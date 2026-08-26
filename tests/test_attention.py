"""Tests for DARKformer attention modules."""

import pytest
import torch

import darkformer.attention as attention_module
from darkformer.attention import (
    CrossAttention,
    DarkformerKernelAttention,
    RotaryEmbedding,
    SelfAttention,
)


def _feature_attention_reference(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    query_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    weights = query_features @ key_features.transpose(-1, -2)
    if causal:
        query_length = query_features.shape[-2]
        key_length = key_features.shape[-2]
        causal_mask = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=weights.device,
        ).tril()
        weights = weights.masked_fill(~causal_mask, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    output = (weights @ value) / denominator
    if query_mask is not None:
        output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
    return output


def test_noncausal_linear_attention_matches_feature_reference() -> None:
    """Noncausal attention matches an explicit normalized feature kernel."""
    generator = torch.Generator().manual_seed(7)
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=2,
        num_features=16,
        orthogonal_features=False,
        causal=False,
        projection_seed=11,
        deterministic=True,
    )
    query = 0.25 * torch.randn(2, 2, 3, 4, generator=generator)
    key = 0.25 * torch.randn(2, 2, 5, 4, generator=generator)
    value = torch.randn(2, 2, 5, 3, generator=generator)
    query_mask = torch.tensor(
        [[True, False, True], [True, True, False]],
    )
    key_mask = torch.tensor(
        [
            [True, True, False, True, False],
            [True, False, True, True, True],
        ],
    )

    actual = attention(
        query,
        key,
        value,
        query_mask=query_mask,
        key_mask=key_mask,
    )
    query_features, key_features = attention.random_features(
        query,
        key,
        key_mask=key_mask,
    )
    expected = _feature_attention_reference(
        query_features,
        key_features,
        value,
        causal=False,
        query_mask=query_mask,
    )

    assert actual.shape == (2, 2, 3, 3)
    torch.testing.assert_close(actual, expected)


def test_causal_linear_attention_matches_prefix_reference() -> None:
    """Chunked causal attention matches explicit lower-triangular weights."""
    generator = torch.Generator().manual_seed(13)
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=2,
        num_features=16,
        orthogonal_features=False,
        causal=True,
        causal_chunk_size=2,
        projection_seed=17,
        deterministic=True,
    )
    query = 0.2 * torch.randn(1, 2, 5, 4, generator=generator)
    key = 0.2 * torch.randn(1, 2, 5, 4, generator=generator)
    value = torch.randn(1, 2, 5, 3, generator=generator)
    mask = torch.tensor([[True, True, True, False, False]])

    actual = attention(
        query,
        key,
        value,
        query_mask=mask,
        key_mask=mask,
    )
    query_features, key_features = attention.random_features(
        query,
        key,
        key_mask=mask,
    )
    expected = _feature_attention_reference(
        query_features,
        key_features,
        value,
        causal=True,
        query_mask=mask,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, :, 3:]) == 0


def test_exact_mode_matches_transformed_softmax_attention() -> None:
    """Exact mode applies softmax after the learned Mahalanobis transform."""
    generator = torch.Generator().manual_seed(19)
    head_dim = 3
    attention = DarkformerKernelAttention(
        head_dim=head_dim,
        num_heads=2,
        num_features=8,
        attention_mode="exact",
        exact_backend="sdpa",
        deterministic=True,
    )
    geometry = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, -0.1, 1.1]],
            [[0.9, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 0.7]],
        ],
    )
    with torch.no_grad():
        attention.geometry.copy_(geometry)
    query = torch.randn(1, 2, 3, 3, generator=generator)
    key = torch.randn(1, 2, 4, 3, generator=generator)
    value = torch.randn(1, 2, 4, 5, generator=generator)

    actual = attention(query, key, value)
    normalizer = head_dim**-0.25
    transformed_query = torch.einsum(
        "bhnd,hrd->bhnr",
        query * normalizer,
        geometry,
    )
    transformed_key = torch.einsum(
        "bhnd,hrd->bhnr",
        key * normalizer,
        geometry,
    )
    scores = transformed_query @ transformed_key.transpose(-1, -2)
    expected = torch.softmax(scores, dim=-1) @ value

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_auto_mode_routes_around_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode sends short inputs to the exact backend only."""
    calls: list[tuple[int, int]] = []

    def fake_exact_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        calls.append((query.shape[-2], key.shape[-2]))
        shape = (*query.shape[:-1], value.shape[-1])
        return value.new_full(shape, 7.0)

    monkeypatch.setattr(
        attention_module,
        "exact_attention",
        fake_exact_attention,
    )
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
        orthogonal_features=False,
        attention_mode="auto",
        exact_threshold=3,
        deterministic=True,
    )
    short_query = torch.randn(1, 1, 2, 4)
    short_key = torch.randn(1, 1, 3, 4)
    short_value = torch.randn(1, 1, 3, 2)

    short_output = attention(short_query, short_key, short_value)

    assert calls == [(2, 3)]
    torch.testing.assert_close(
        short_output,
        torch.full_like(short_output, 7.0),
    )

    long_query = torch.randn(1, 1, 4, 4)
    long_key = torch.randn(1, 1, 5, 4)
    long_value = torch.randn(1, 1, 5, 2)
    long_output = attention(long_query, long_key, long_value)

    assert calls == [(2, 3)]
    assert long_output.shape == (1, 1, 4, 2)


def test_exact_mode_does_not_redraw_unused_features() -> None:
    """Exact attention does not advance the random-feature redraw schedule."""
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
        attention_mode="exact",
        exact_backend="sdpa",
        feature_redraw_interval=1,
        projection_seed=5,
    )
    tensor = torch.randn(1, 1, 3, 4)
    projection = attention.random_features.projection_matrix.clone()

    attention(tensor, tensor, tensor)

    torch.testing.assert_close(
        attention.random_features.projection_matrix,
        projection,
        rtol=0,
        atol=0,
    )
    assert attention._calls_since_redraw.item() == 0


def test_unseeded_redraw_is_reproducible_from_checkpoint_state() -> None:
    """Replicas redraw alike after their buffers are synchronized."""
    first = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
    )
    second = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
    )
    second.load_state_dict(first.state_dict())

    first.redraw_projection_matrices_()
    torch.randn(32)
    second.redraw_projection_matrices_()

    torch.testing.assert_close(
        first.random_features.projection_matrix,
        second.random_features.projection_matrix,
        rtol=0,
        atol=0,
    )


def test_self_attention_shape_and_gradients() -> None:
    """Self-attention preserves shape and differentiates all core inputs."""
    torch.manual_seed(23)
    attention = SelfAttention(
        8,
        heads=2,
        head_dim=4,
        num_features=12,
        orthogonal_features=False,
        causal=True,
        causal_chunk_size=2,
        projection_seed=29,
        deterministic=True,
    )
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]],
    )

    output = attention(inputs, mask=mask)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert torch.count_nonzero(output[~mask]) == 0
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) > 0
    assert attention.geometry.grad is not None
    assert torch.count_nonzero(attention.geometry.grad) > 0


def test_cross_attention_unequal_lengths_shape_and_gradients() -> None:
    """Cross-attention supports distinct query and context lengths."""
    torch.manual_seed(31)
    attention = CrossAttention(
        8,
        context_dim=6,
        heads=2,
        head_dim=4,
        num_features=12,
        orthogonal_features=False,
        projection_seed=37,
        deterministic=True,
    )
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 5, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    context_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, True],
        ],
    )

    output = attention(
        inputs,
        context,
        mask=mask,
        context_mask=context_mask,
    )
    output.square().mean().backward()

    assert output.shape == (2, 3, 8)
    assert torch.count_nonzero(output[~mask]) == 0
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) > 0
    assert context.grad is not None
    assert torch.count_nonzero(context.grad) > 0
    assert attention.geometry.grad is not None
    assert torch.count_nonzero(attention.geometry.grad) > 0


def test_causal_state_matches_full_attention_across_chunks() -> None:
    """Recurrent causal attention matches a full-sequence evaluation."""
    generator = torch.Generator().manual_seed(41)
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=2,
        num_features=16,
        orthogonal_features=False,
        causal=True,
        causal_chunk_size=2,
        projection_seed=43,
        deterministic=True,
    )
    query = 0.25 * torch.randn(2, 2, 7, 4, generator=generator)
    key = 0.25 * torch.randn(2, 2, 7, 4, generator=generator)
    value = torch.randn(2, 2, 7, 3, generator=generator)
    mask = torch.tensor(
        [
            [True, True, True, True, False, True, True],
            [True, True, False, True, True, True, False],
        ]
    )

    expected = attention(
        query,
        key,
        value,
        query_mask=mask,
        key_mask=mask,
    )
    first, state = attention.forward_with_state(
        query[:, :, :3],
        key[:, :, :3],
        value[:, :, :3],
        query_mask=mask[:, :3],
        key_mask=mask[:, :3],
    )
    second, state = attention.forward_with_state(
        query[:, :, 3:5],
        key[:, :, 3:5],
        value[:, :, 3:5],
        state=state,
        query_mask=mask[:, 3:5],
        key_mask=mask[:, 3:5],
    )
    third, state = attention.forward_with_state(
        query[:, :, 5:],
        key[:, :, 5:],
        value[:, :, 5:],
        state=state,
        query_mask=mask[:, 5:],
        key_mask=mask[:, 5:],
    )

    torch.testing.assert_close(torch.cat((first, second, third), dim=2), expected)
    assert state.sequence_length == 7


def test_causal_attention_does_not_depend_on_future_keys() -> None:
    """Changing future keys and values leaves earlier outputs unchanged."""
    attention = DarkformerKernelAttention(
        head_dim=2,
        num_heads=1,
        num_features=4,
        orthogonal_features=False,
        causal=True,
        causal_chunk_size=8,
        eps=1e-2,
        projection_seed=53,
        deterministic=True,
    )
    with torch.no_grad():
        attention.random_features.projection_matrix.copy_(
            torch.tensor([[20.0, 0.0], [-20.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )
    query = torch.tensor([[[[1.0, 0.0], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]])
    value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]])
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, 2] = torch.tensor([10.0, 0.0])
    changed_value[:, :, 2:] *= -7.0

    original = attention(query, key, value)
    changed = attention(query, changed_key, changed_value)

    torch.testing.assert_close(
        original[:, :, :2],
        changed[:, :, :2],
        rtol=0,
        atol=0,
    )


def test_all_masked_causal_chunk_preserves_statistics() -> None:
    """Masked continuation positions do not alter recurrent statistics."""
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
        causal=True,
        deterministic=True,
    )
    tensor = torch.randn(1, 1, 2, 4)
    _, state = attention.forward_with_state(tensor, tensor, tensor)
    masked = torch.zeros(1, 3, dtype=torch.bool)
    continuation = torch.randn(1, 1, 3, 4)

    output, next_state = attention.forward_with_state(
        continuation,
        continuation,
        continuation,
        state=state,
        query_mask=masked,
        key_mask=masked,
    )

    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(next_state.key_sum, state.key_sum)
    torch.testing.assert_close(next_state.key_value_sum, state.key_value_sum)
    torch.testing.assert_close(next_state.key_log_scale, state.key_log_scale)
    assert next_state.sequence_length == 5


def test_causal_state_rejects_projection_redraw() -> None:
    """A state cannot be continued after its feature basis changes."""
    attention = DarkformerKernelAttention(
        head_dim=4,
        num_heads=1,
        num_features=8,
        causal=True,
        deterministic=True,
    )
    tensor = torch.randn(1, 1, 2, 4)
    _, state = attention.forward_with_state(tensor, tensor, tensor)
    attention.redraw_projection_matrices_(force=True)

    with pytest.raises(RuntimeError, match="projection matrices changed"):
        attention.forward_with_state(
            tensor[:, :, :1],
            tensor[:, :, :1],
            tensor[:, :, :1],
            state=state,
        )


def test_context_state_matches_cross_attention() -> None:
    """Precomputed context statistics match repeated cross-attention."""
    generator = torch.Generator().manual_seed(59)
    attention = CrossAttention(
        8,
        context_dim=6,
        heads=2,
        head_dim=4,
        num_features=16,
        orthogonal_features=False,
        projection_seed=61,
        deterministic=True,
    )
    inputs = torch.randn(2, 5, 8, generator=generator)
    context = torch.randn(2, 7, 6, generator=generator)
    mask = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )
    context_mask = torch.tensor(
        [
            [True, True, True, False, True, False, True],
            [True, False, True, True, True, True, False],
        ]
    )

    expected = attention(
        inputs,
        context,
        mask=mask,
        context_mask=context_mask,
    )
    state = attention.build_context_state(context, context_mask=context_mask)
    first = attention.forward_with_state(inputs[:, :2], state, mask=mask[:, :2])
    second = attention.forward_with_state(inputs[:, 2:], state, mask=mask[:, 2:])

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)


def test_self_attention_state_uses_rotary_offset() -> None:
    """Stateful self-attention preserves full-sequence rotary positions."""
    torch.manual_seed(67)
    attention = SelfAttention(
        8,
        heads=2,
        head_dim=4,
        num_features=16,
        causal=True,
        rotary=True,
        dropout=0.0,
        projection_seed=71,
        deterministic=True,
    )
    inputs = torch.randn(2, 6, 8)

    expected = attention(inputs)
    first, state = attention.forward_with_state(inputs[:, :4])
    second, state = attention.forward_with_state(inputs[:, 4:], state=state)

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    assert state.sequence_length == 6


def test_rotary_offset_matches_full_sequence_for_odd_head_dim() -> None:
    """Rotary chunks match the corresponding full-sequence positions."""
    rotary = RotaryEmbedding(5)
    query = torch.randn(1, 2, 6, 5)
    key = torch.randn(1, 2, 6, 5)

    full_query, full_key = rotary(query, key)
    first_query, first_key = rotary(query[:, :, :2], key[:, :, :2])
    second_query, second_key = rotary(
        query[:, :, 2:],
        key[:, :, 2:],
        offset=2,
    )

    torch.testing.assert_close(
        torch.cat((first_query, second_query), dim=2),
        full_query,
    )
    torch.testing.assert_close(torch.cat((first_key, second_key), dim=2), full_key)


def test_rotary_cache_rebuilds_after_dtype_round_trip() -> None:
    """Rotary values retain precision across dtype changes."""
    rotary = RotaryEmbedding(8)
    query = torch.randn(1, 2, 32, 8)
    key = torch.randn(1, 2, 32, 8)
    expected_query, expected_key = rotary(query, key)

    rotary.to(dtype=torch.bfloat16)
    rotary(query.bfloat16(), key.bfloat16())
    rotary.float()
    actual_query, actual_key = rotary(query, key)

    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)


def test_bfloat16_linear_attention_returns_model_dtype() -> None:
    """Linear attention accumulates stably and returns the requested dtype."""
    attention = SelfAttention(
        8,
        heads=2,
        head_dim=4,
        num_features=8,
        causal=True,
        deterministic=True,
    ).to(dtype=torch.bfloat16)
    inputs = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    output = attention(inputs)

    assert output.dtype == torch.bfloat16
    assert torch.all(torch.isfinite(output))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward() -> None:
    """Modules move to CUDA with their parameters and projection buffers."""
    attention = SelfAttention(
        16,
        heads=2,
        head_dim=8,
        num_features=16,
        causal=True,
        deterministic=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(
        2,
        8,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = attention(inputs)
    output.float().square().mean().backward()

    assert output.device.type == "cuda"
    assert output.dtype == torch.bfloat16
    assert inputs.grad is not None
    assert attention.geometry.grad is not None
