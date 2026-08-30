"""Tests for data-aware positive random features."""

import math

import pytest
import torch

from darkformer_pytorch.random_features import (
    DataAwareRandomFeatures,
    _gaussian_projection,
)


def test_identity_covariance_initialization() -> None:
    """The default full-rank geometry starts at the identity."""
    head_dim = 4
    num_heads = 3
    random_features = DataAwareRandomFeatures(
        head_dim=head_dim,
        num_heads=num_heads,
        num_features=16,
    )

    expected = torch.eye(head_dim).expand(num_heads, -1, -1)
    torch.testing.assert_close(random_features.covariance(), expected)
    assert isinstance(random_features.geometry, torch.nn.Parameter)
    assert "projection_matrix" in dict(random_features.named_buffers())


def test_variance_reduced_feature_defaults() -> None:
    """The default estimator uses orthogonal features without an additive floor."""
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=16,
    )

    assert random_features.orthogonal
    assert random_features.eps == 0.0


def test_orthogonal_projection_is_invariant_to_qr_sign_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The projection is invariant to an equivalent QR sign convention."""
    seed = 3
    baseline = _gaussian_projection(
        5,
        3,
        orthogonal=True,
        generator=torch.Generator().manual_seed(seed),
    )
    original_qr = torch.linalg.qr

    def fake_qr(
        unstructured: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert mode == "reduced"
        orthogonal, upper = original_qr(unstructured, mode="reduced")
        signs = unstructured.new_tensor([-1.0, 1.0, -1.0])
        return orthogonal * signs, signs[:, None] * upper

    monkeypatch.setattr(torch.linalg, "qr", fake_qr)
    flipped = _gaussian_projection(
        5,
        3,
        orthogonal=True,
        generator=torch.Generator().manual_seed(seed),
    )

    torch.testing.assert_close(flipped, baseline, rtol=0, atol=0)


def test_default_whitening_uses_temperature_preserving_scale() -> None:
    """Low-level whitening uses the high-level policy's geometry scale."""
    generator = torch.Generator().manual_seed(5)
    samples = torch.randn(8, 2, 128, 3, generator=generator)
    mixing = torch.tensor(
        [[2.0, 0.0, 0.0], [0.7, 0.5, 0.0], [-0.2, 0.4, 1.5]],
    )
    samples = samples @ mixing.transpose(0, 1)
    random_features = DataAwareRandomFeatures(
        head_dim=3,
        num_heads=2,
        num_features=16,
    )

    random_features.initialize_whitening_(samples, regularization=0.0)
    transformed = samples @ random_features.geometry.transpose(-1, -2)
    centered = transformed - transformed.mean(dim=(0, 2), keepdim=True)
    covariance = torch.einsum("bhld,bhle->hde", centered, centered)
    covariance = covariance / (samples.shape[0] * samples.shape[2] - 1)

    geometry_scale = 3**-0.25
    expected = geometry_scale**2 * torch.eye(3).expand(2, -1, -1)
    torch.testing.assert_close(covariance, expected, rtol=2e-4, atol=2e-4)


def test_whitening_geometry_scale_controls_transformed_covariance() -> None:
    """Post-whitening scaling sets covariance without changing its shape."""
    generator = torch.Generator().manual_seed(6)
    samples = torch.randn(8, 2, 128, 3, generator=generator)
    mixing = torch.tensor(
        [[2.0, 0.0, 0.0], [0.7, 0.5, 0.0], [-0.2, 0.4, 1.5]],
    )
    samples = samples @ mixing.transpose(0, 1)
    random_features = DataAwareRandomFeatures(3, 2, 16)
    geometry_scale = 0.5

    with pytest.warns(
        UserWarning,
        match=r"geometry_scale=0\.5.*approximately 0\.75",
    ):
        random_features.initialize_whitening_(
            samples,
            regularization=0.0,
            geometry_scale=geometry_scale,
        )
    transformed = samples @ random_features.geometry.transpose(-1, -2)
    centered = transformed - transformed.mean(dim=(0, 2), keepdim=True)
    covariance = torch.einsum("bhld,bhle->hde", centered, centered)
    covariance = covariance / (samples.shape[0] * samples.shape[2] - 1)

    expected = geometry_scale**2 * torch.eye(3).expand(2, -1, -1)
    torch.testing.assert_close(covariance, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("geometry_scale", [0.0, -0.5, math.nan, math.inf])
def test_whitening_rejects_invalid_geometry_scale(geometry_scale: float) -> None:
    """Scaling must preserve a finite, full-rank whitening geometry."""
    random_features = DataAwareRandomFeatures(3, 1, 16)

    with pytest.raises(ValueError, match="geometry_scale must be finite and positive"):
        random_features.initialize_whitening_(
            torch.randn(2, 1, 4, 3),
            geometry_scale=geometry_scale,
        )


def test_variance_optimal_proposal_matches_theorem_3_2() -> None:
    """The separate proposal uses (I + 2 Lambda)(I - 2 Lambda)^-1."""
    generator = torch.Generator().manual_seed(8)
    head_dim = 4
    samples = 0.3 * torch.randn(4, 2, 32, head_dim, generator=generator)
    random_features = DataAwareRandomFeatures(
        head_dim=head_dim,
        num_heads=2,
        num_features=16,
        projection_seed=9,
    )
    with torch.no_grad():
        random_features.geometry[0].copy_(
            torch.diag(torch.tensor([0.8, 1.0, 1.1, 0.9]))
        )
        random_features.geometry[1].copy_(
            torch.tensor(
                [
                    [1.0, 0.1, 0.0, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                    [0.0, 0.0, 1.1, 0.1],
                    [0.1, 0.0, 0.0, 0.8],
                ]
            )
        )

    random_features.initialize_variance_optimal_proposal_(samples)

    transformed = torch.matmul(
        samples * head_dim**-0.25,
        random_features.geometry.transpose(-1, -2),
    )
    centered = transformed - transformed.mean(dim=(0, 2), keepdim=True)
    covariance = torch.einsum("bhld,bhle->hde", centered, centered)
    covariance = covariance / (samples.shape[0] * samples.shape[2] - 1)
    identity = torch.eye(head_dim).expand(2, -1, -1)
    expected = torch.linalg.solve(
        identity - 2.0 * covariance,
        identity + 2.0 * covariance,
    )

    assert random_features.proposal_is_active
    torch.testing.assert_close(
        random_features.proposal_covariance(),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


def test_variance_optimal_proposal_supports_low_rank_geometry() -> None:
    """The proposal is full-density in the geometry's feature coordinates."""
    with pytest.warns(UserWarning, match="importance-sampling interpretation"):
        random_features = DataAwareRandomFeatures(
            head_dim=4,
            num_heads=2,
            num_features=16,
            rank=2,
            projection_seed=10,
        )
    samples = 0.1 * torch.randn(4, 2, 16, 4)

    random_features.initialize_variance_optimal_proposal_(samples)

    assert random_features.proposal_is_active
    assert random_features.proposal_covariance().shape == (2, 2, 2)


def test_variance_optimal_proposal_preserves_kernel_expectation() -> None:
    """Density-ratio features remain unbiased for the configured kernel."""
    random_features = DataAwareRandomFeatures(
        head_dim=1,
        num_heads=1,
        num_features=65_536,
        orthogonal=False,
        projection_seed=31,
    )
    calibration_value = math.sqrt(0.125)
    calibration = torch.tensor(
        [[[[-calibration_value], [calibration_value]]]],
    )
    random_features.initialize_variance_optimal_proposal_(calibration)
    query = torch.tensor([[[[0.2]]]])
    key = torch.tensor([[[[0.1]]]])

    query_features, key_features = random_features(
        query,
        key,
        stabilize=False,
    )

    estimate = (query_features * key_features).sum()
    expected = torch.exp(query.flatten() @ key.flatten())
    torch.testing.assert_close(estimate, expected, rtol=0.01, atol=0.01)


def test_variance_optimal_proposal_can_reset_to_isotropic_sampling() -> None:
    """Resetting removes the proposal and its derived caches."""
    random_features = DataAwareRandomFeatures(2, 2, 16, projection_seed=37)
    calibration = 0.1 * torch.randn(4, 2, 8, 2)
    random_features.initialize_variance_optimal_proposal_(calibration)

    returned = random_features.reset_variance_optimal_proposal_()

    assert returned is random_features
    assert not random_features.proposal_is_active
    assert random_features._proposal_projection.numel() == 0
    assert random_features._proposal_log_weights.numel() == 0
    expected = torch.eye(2).expand(2, -1, -1)
    torch.testing.assert_close(random_features.proposal_covariance(), expected)


def test_variance_optimal_proposal_rejects_nonnormalizable_covariance() -> None:
    """The closed-form proposal exists only below the half-variance boundary."""
    random_features = DataAwareRandomFeatures(1, 1, 16)
    calibration = torch.tensor([[[[-0.5], [0.5]]]])

    with pytest.raises(ValueError, match=r"eigenvalue to be below 0\.5"):
        random_features.initialize_variance_optimal_proposal_(calibration)


@pytest.mark.filterwarnings(
    "ignore:literal whitening targets unit transformed covariance"
)
def test_whitening_initialization_respects_masks() -> None:
    """Masked samples do not contribute to covariance calibration."""
    generator = torch.Generator().manual_seed(7)
    samples = torch.randn(1, 1, 8, 3, generator=generator)
    samples[:, :, 6:] = 1_000.0
    mask = torch.tensor([[True, True, True, True, True, True, False, False]])
    masked = DataAwareRandomFeatures(3, 1, 16)
    sliced = DataAwareRandomFeatures(3, 1, 16)

    masked.initialize_whitening_(samples, query_mask=mask)
    sliced.initialize_whitening_(samples[:, :, :6])

    torch.testing.assert_close(masked.geometry, sliced.geometry)


@pytest.mark.filterwarnings(
    "ignore:literal whitening targets unit transformed covariance"
)
def test_shared_whitening_pools_head_means() -> None:
    """Shared geometry uses covariance across every head sample."""
    samples = torch.tensor(
        [
            [
                [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
                [[9.0, 0.0], [11.0, 0.0], [10.0, -1.0], [10.0, 1.0]],
            ]
        ]
    )
    random_features = DataAwareRandomFeatures(
        head_dim=2,
        num_heads=2,
        num_features=16,
        per_head=False,
    )

    random_features.initialize_whitening_(samples, regularization=0.0)
    transformed = samples @ random_features.geometry.transpose(-1, -2)
    centered = transformed - transformed.mean(dim=(0, 1, 2), keepdim=True)
    covariance = torch.einsum("bhld,bhle->de", centered, centered)
    covariance = covariance / (samples.numel() // samples.shape[-1] - 1)

    geometry_scale = 2**-0.25
    torch.testing.assert_close(
        covariance,
        geometry_scale**2 * torch.eye(2),
        rtol=2e-4,
        atol=2e-4,
    )


def test_low_rank_geometry_warns_and_rejects_whitening() -> None:
    """Singular geometry is explicit about its theoretical limitation."""
    with pytest.warns(UserWarning, match="importance-sampling interpretation"):
        random_features = DataAwareRandomFeatures(
            head_dim=4,
            num_heads=1,
            num_features=16,
            rank=2,
        )

    with pytest.raises(ValueError, match="full-rank geometry"):
        random_features.initialize_whitening_(torch.randn(2, 1, 8, 4))


def test_covariance_is_positive_semidefinite() -> None:
    """A low-rank learned geometry always produces a PSD covariance."""
    torch.manual_seed(11)
    with pytest.warns(UserWarning, match="importance-sampling interpretation"):
        random_features = DataAwareRandomFeatures(
            head_dim=5,
            num_heads=2,
            num_features=16,
            rank=3,
        )
    with torch.no_grad():
        random_features.geometry.normal_()

    covariance = random_features.covariance()
    torch.testing.assert_close(
        covariance,
        covariance.transpose(-1, -2),
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert torch.all(eigenvalues >= -1e-5)


def test_forward_shapes_and_values_are_finite() -> None:
    """Cross-attention inputs produce positive features of the right shape."""
    torch.manual_seed(13)
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=3,
        num_features=32,
    )
    query = 0.25 * torch.randn(2, 3, 5, 4)
    key = 0.25 * torch.randn(2, 3, 7, 4)

    query_features, key_features = random_features(query, key)

    assert query_features.shape == (2, 3, 5, 32)
    assert key_features.shape == (2, 3, 7, 32)
    assert torch.all(torch.isfinite(query_features))
    assert torch.all(torch.isfinite(key_features))
    assert torch.all(query_features > 0)
    assert torch.all(key_features > 0)


@pytest.mark.parametrize("eps", [0.0, 1e-4])
def test_stabilization_keeps_all_negative_logits_representable(eps: float) -> None:
    """Large negative logits neither underflow nor overflow after stabilization."""
    random_features = DataAwareRandomFeatures(
        head_dim=2,
        num_heads=1,
        num_features=4,
        eps=eps,
        deterministic=True,
    )
    with torch.no_grad():
        random_features.geometry.copy_(30.0 * torch.eye(2).unsqueeze(0))
        random_features.projection_matrix.zero_()
    query = torch.ones(1, 1, 3, 2)
    key = torch.ones(1, 1, 4, 2)

    query_features, key_features = random_features(query, key)

    assert torch.all(torch.isfinite(query_features))
    assert torch.all(torch.isfinite(key_features))
    assert torch.all(query_features > 0.0)
    assert torch.all(key_features > 0.0)


def test_key_mask_zeros_only_invalid_key_features() -> None:
    """A true mask entry keeps a key and a false entry removes it."""
    torch.manual_seed(17)
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=24,
    )
    query = 0.25 * torch.randn(2, 2, 3, 4)
    key = 0.25 * torch.randn(2, 2, 4, 4)
    key_mask = torch.tensor(
        [[True, False, True, False], [False, True, True, True]],
    )

    unmasked_query, unmasked_key = random_features(
        query,
        key,
        stabilize=False,
    )
    masked_query, masked_key = random_features(
        query,
        key,
        key_mask=key_mask,
        stabilize=False,
    )

    torch.testing.assert_close(masked_query, unmasked_query)
    expanded_mask = key_mask[:, None, :, None].expand_as(masked_key)
    torch.testing.assert_close(
        masked_key[expanded_mask],
        unmasked_key[expanded_mask],
    )
    torch.testing.assert_close(
        masked_key[~expanded_mask],
        torch.zeros_like(masked_key[~expanded_mask]),
    )


def test_projection_is_deterministic_until_redraw() -> None:
    """Feature values change only when the projection buffer is redrawn."""
    torch.manual_seed(19)
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=32,
    )
    query = 0.2 * torch.randn(1, 2, 3, 4)
    key = 0.2 * torch.randn(1, 2, 3, 4)

    first_query, first_key = random_features(
        query,
        key,
        stabilize=False,
    )
    second_query, second_key = random_features(
        query,
        key,
        stabilize=False,
    )
    torch.testing.assert_close(first_query, second_query, rtol=0, atol=0)
    torch.testing.assert_close(first_key, second_key, rtol=0, atol=0)

    original_projection = random_features.projection_matrix.clone()
    generator = torch.Generator().manual_seed(23)
    random_features.redraw_projection_(generator=generator)
    assert not torch.equal(
        original_projection,
        random_features.projection_matrix,
    )

    redrawn_query, redrawn_key = random_features(
        query,
        key,
        stabilize=False,
    )
    assert not torch.equal(first_query, redrawn_query)
    assert not torch.equal(first_key, redrawn_key)

    repeated_query, repeated_key = random_features(
        query,
        key,
        stabilize=False,
    )
    torch.testing.assert_close(redrawn_query, repeated_query, rtol=0, atol=0)
    torch.testing.assert_close(redrawn_key, repeated_key, rtol=0, atol=0)


def test_projection_seed_is_independent_of_global_rng() -> None:
    """A projection seed reproduces initialization across RNG states."""
    torch.manual_seed(37)
    first = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=32,
        projection_seed=41,
    )
    torch.manual_seed(43)
    second = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=32,
        projection_seed=41,
    )

    torch.testing.assert_close(
        first.projection_matrix,
        second.projection_matrix,
        rtol=0,
        atol=0,
    )


def test_fixed_projection_requires_force_or_unfix_to_redraw() -> None:
    """Fixed projections ignore redraws unless explicitly overridden."""
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=32,
        projection_seed=47,
        deterministic=True,
    )
    initial_projection = random_features.projection_matrix.clone()

    generator = torch.Generator().manual_seed(53)
    random_features.redraw_projection_(generator=generator)
    torch.testing.assert_close(
        random_features.projection_matrix,
        initial_projection,
        rtol=0,
        atol=0,
    )

    force_generator = torch.Generator().manual_seed(59)
    random_features.redraw_projection_(
        generator=force_generator,
        force=True,
    )
    forced_projection = random_features.projection_matrix.clone()
    assert not torch.equal(forced_projection, initial_projection)

    random_features.fix_projection_matrix_()
    random_features.unfix_projection_matrix_()
    redraw_generator = torch.Generator().manual_seed(61)
    random_features.redraw_projection_(generator=redraw_generator)
    assert not torch.equal(
        random_features.projection_matrix,
        forced_projection,
    )


def test_projection_fixed_cache_is_restored_from_state_dict() -> None:
    """Checkpoint loading synchronizes the no-host-sync lifecycle cache."""
    fixed = DataAwareRandomFeatures(4, 1, 8, deterministic=True)
    restored = DataAwareRandomFeatures(4, 1, 8, deterministic=False)

    restored.load_state_dict(fixed.state_dict())

    assert restored.projection_is_fixed


def test_proposal_caches_refresh_after_state_load_and_redraw() -> None:
    """Derived proposal samples follow the restored and redrawn base samples."""
    calibration_value = math.sqrt(0.125)
    calibration = torch.tensor(
        [[[[-calibration_value], [calibration_value]]]],
    )
    original = DataAwareRandomFeatures(1, 1, 32, projection_seed=67)
    original.initialize_variance_optimal_proposal_(calibration)
    restored = DataAwareRandomFeatures(1, 1, 32, projection_seed=71)

    restored.load_state_dict(original.state_dict())

    assert restored.proposal_is_active
    torch.testing.assert_close(
        restored._proposal_projection,
        original._proposal_projection,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored._proposal_log_weights,
        original._proposal_log_weights,
        rtol=0,
        atol=0,
    )
    cached_projection = restored._proposal_projection.clone()

    restored.redraw_projection_(generator=torch.Generator().manual_seed(73))

    assert not torch.equal(restored._proposal_projection, cached_projection)
    expected_projection = torch.matmul(
        restored.projection_matrix.unsqueeze(0),
        restored._proposal_root.transpose(-1, -2),
    )
    torch.testing.assert_close(
        restored._proposal_projection,
        expected_projection,
        rtol=0,
        atol=0,
    )


def test_proposal_log_weight_precision_survives_module_casts() -> None:
    """Derived density weights keep their accumulation dtype after casting."""
    calibration_value = math.sqrt(0.125)
    calibration = torch.tensor(
        [[[[-calibration_value], [calibration_value]]]],
    )
    random_features = DataAwareRandomFeatures(1, 1, 32, projection_seed=77)
    random_features.initialize_variance_optimal_proposal_(calibration)

    random_features.half()

    assert random_features._proposal_projection.dtype == torch.float16
    assert random_features._proposal_log_weights.dtype == torch.float32

    random_features.redraw_projection_(generator=torch.Generator().manual_seed(78))

    assert random_features._proposal_log_weights.dtype == torch.float32


def test_stabilized_bfloat16_features_preserve_activation_dtype() -> None:
    """Stable fp32 logits do not force feature-sized activations to fp32."""
    generator = torch.Generator().manual_seed(81)
    random_features = DataAwareRandomFeatures(
        4,
        2,
        32,
        orthogonal=False,
        projection_seed=83,
    ).bfloat16()
    query = (0.5 * torch.randn(2, 2, 5, 4, generator=generator)).bfloat16()
    key = (0.5 * torch.randn(2, 2, 7, 4, generator=generator)).bfloat16()

    query_features, key_features = random_features(query, key)

    assert query_features.dtype == torch.bfloat16
    assert key_features.dtype == torch.bfloat16
    assert torch.all(torch.isfinite(query_features))
    assert torch.all(torch.isfinite(key_features))

    # The correction and maxima remain fp32; only the already-stabilized,
    # nonpositive exponent is evaluated in the activation dtype.
    query_logits = random_features.feature_logits(query)
    key_logits = random_features.feature_logits(key)
    assert query_logits.dtype == torch.float32
    assert key_logits.dtype == torch.float32
    expected_query = (
        torch.exp(query_logits - query_logits.amax(dim=-1, keepdim=True))
        * random_features.num_features**-0.5
    )
    expected_key = (
        torch.exp(key_logits - key_logits.amax(dim=(-2, -1), keepdim=True))
        * random_features.num_features**-0.5
    )
    torch.testing.assert_close(
        query_features.float(),
        expected_query,
        rtol=2e-2,
        atol=2e-4,
    )
    torch.testing.assert_close(
        key_features.float(),
        expected_key,
        rtol=2e-2,
        atol=2e-4,
    )


def test_checkpoint_without_proposal_state_loads_as_isotropic() -> None:
    """Checkpoints predating separate proposals retain their original behavior."""
    original = DataAwareRandomFeatures(4, 2, 16, projection_seed=79)
    state_dict = original.state_dict()
    del state_dict["_proposal_root"]
    del state_dict["_proposal_active"]
    restored = DataAwareRandomFeatures(4, 2, 16, projection_seed=83)

    restored.load_state_dict(state_dict)

    assert not restored.proposal_is_active
    expected = torch.eye(4).expand(2, -1, -1)
    torch.testing.assert_close(restored.proposal_covariance(), expected)


def test_geometry_receives_finite_nonzero_gradients() -> None:
    """The reparameterized samples remain differentiable in geometry."""
    torch.manual_seed(29)
    random_features = DataAwareRandomFeatures(
        head_dim=4,
        num_heads=2,
        num_features=64,
    )
    query = 0.2 * torch.randn(2, 2, 3, 4)
    key = 0.2 * torch.randn(2, 2, 3, 4)

    query_features, key_features = random_features(
        query,
        key,
        stabilize=False,
    )
    loss = query_features.square().mean() + key_features.square().mean()
    loss.backward()

    gradient = random_features.geometry.grad
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient))
    assert torch.count_nonzero(gradient) > 0
    assert not random_features.projection_matrix.requires_grad


def test_iid_features_converge_to_mahalanobis_kernel() -> None:
    """A large iid feature sample approaches the DARKformer kernel."""
    torch.manual_seed(31)
    head_dim = 3
    random_features = DataAwareRandomFeatures(
        head_dim=head_dim,
        num_heads=1,
        num_features=65_536,
        orthogonal=False,
    )
    geometry = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, -0.1, 1.1]],
    )
    with torch.no_grad():
        random_features.geometry[0].copy_(geometry)

    query = torch.tensor([[[[0.4, -0.2, 0.3]]]])
    key = torch.tensor([[[[-0.1, 0.25, 0.2]]]])
    query_features, key_features = random_features(
        query,
        key,
        stabilize=False,
    )

    estimate = (query_features * key_features).sum(dim=-1).squeeze()
    covariance = random_features.covariance()[0]
    exponent = query.flatten() @ covariance @ key.flatten()
    expected = torch.exp(exponent / math.sqrt(head_dim))
    torch.testing.assert_close(estimate, expected, rtol=0.03, atol=0.02)
