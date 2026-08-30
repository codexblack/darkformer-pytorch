"""Tests for DARKformer language and encoder-decoder models."""

from typing import cast
from unittest import mock

import pytest
import torch
from torch import nn

from darkformer_pytorch.attention import AttentionMode, DarkformerKernelAttention
from darkformer_pytorch.model import (
    Darkformer,
    DarkformerBlock,
    DarkformerEncDec,
    DarkformerLM,
    _sample_next_token,
)
from darkformer_pytorch.random_features import DataAwareRandomFeatures

ProjectionModel = DarkformerLM | DarkformerEncDec

pytestmark = pytest.mark.filterwarnings(
    "ignore:literal whitening targets unit transformed covariance"
)


def _language_model(
    *,
    depth: int = 1,
    deterministic: bool = True,
    fixed_projection: bool | None = None,
    backend_deterministic: bool | None = None,
    causal: bool = True,
    attention_mode: AttentionMode = "linear",
    rotary: bool = False,
    max_seq_len: int = 10,
) -> DarkformerLM:
    return DarkformerLM(
        vocab_size=19,
        dim=8,
        depth=depth,
        heads=2,
        head_dim=4,
        num_features=8,
        mlp_dim=16,
        max_seq_len=max_seq_len,
        causal=causal,
        attention_mode=attention_mode,
        dropout=0.0,
        rotary=rotary,
        causal_chunk_size=2,
        projection_seed=7,
        fixed_projection=fixed_projection,
        backend_deterministic=backend_deterministic,
        deterministic=deterministic,
    )


def _encoder_decoder(
    *,
    depth: int = 1,
    deterministic: bool = True,
    attention_mode: AttentionMode = "linear",
    rotary: bool = False,
    dropout: float = 0.0,
) -> DarkformerEncDec:
    return DarkformerEncDec(
        source_vocab_size=17,
        target_vocab_size=19,
        dim=8,
        depth=depth,
        heads=2,
        head_dim=4,
        num_features=8,
        mlp_dim=16,
        max_source_length=8,
        max_target_length=8,
        attention_mode=attention_mode,
        dropout=dropout,
        rotary=rotary,
        causal_chunk_size=2,
        projection_seed=23,
        deterministic=deterministic,
    )


def _random_feature_modules(
    model: nn.Module,
) -> list[DataAwareRandomFeatures]:
    return [
        module
        for module in model.modules()
        if isinstance(module, DataAwareRandomFeatures)
    ]


def _projection_snapshots(model: nn.Module) -> list[torch.Tensor]:
    return [
        module.projection_matrix.detach().clone()
        for module in _random_feature_modules(model)
    ]


def _assert_projection_controls(
    model: ProjectionModel,
    *,
    expected_modules: int,
) -> None:
    feature_modules = _random_feature_modules(model)
    assert len(feature_modules) == expected_modules
    initial = _projection_snapshots(model)

    model.fix_projection_matrices_()
    assert all(module.projection_is_fixed for module in feature_modules)
    model.redraw_projection_matrices_()
    fixed = _projection_snapshots(model)
    assert all(
        torch.equal(before, after) for before, after in zip(initial, fixed, strict=True)
    )

    model.redraw_projection_matrices_(force=True)
    forced = _projection_snapshots(model)
    assert all(
        not torch.equal(before, after)
        for before, after in zip(fixed, forced, strict=True)
    )

    model.unfix_projection_matrices_()
    assert all(not module.projection_is_fixed for module in feature_modules)
    model.redraw_projection_matrices_()
    redrawn = _projection_snapshots(model)
    assert all(
        not torch.equal(before, after)
        for before, after in zip(forced, redrawn, strict=True)
    )


def test_sample_next_token_top_k_keeps_exactly_k_tied_candidates() -> None:
    logits = torch.tensor(
        [[2.0, 2.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    )

    with mock.patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
        sampled = _sample_next_token(logits, temperature=1.0, top_k=2)

    probabilities = cast(torch.Tensor, multinomial.call_args.args[0])
    assert sampled.shape == (2, 1)
    assert torch.count_nonzero(probabilities, dim=-1).tolist() == [2, 2]


@pytest.mark.parametrize("shape", [(4,), (2, 1, 4)])
def test_sample_next_token_rejects_non_matrix_logits(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape \(batch, vocab_size\)"):
        _sample_next_token(torch.zeros(shape), temperature=1.0, top_k=None)


@torch.no_grad()
def _full_prefix_language_generation(
    model: DarkformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    generated = prompt
    try:
        for _ in range(max_new_tokens):
            next_token = model(generated)[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return generated


@torch.no_grad()
def _full_prefix_encoder_decoder_generation(
    model: DarkformerEncDec,
    source_tokens: torch.Tensor,
    prompt: torch.Tensor,
    max_new_tokens: int,
    source_mask: torch.Tensor | None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    generated = prompt
    try:
        context = model.encode(source_tokens, source_mask=source_mask)
        for _ in range(max_new_tokens):
            next_token = model.decode(
                generated,
                context,
                source_mask=source_mask,
            )[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)
    return generated


def test_language_model_logits_loss_and_generation() -> None:
    """The language model trains and extends prompts on CPU."""
    torch.manual_seed(29)
    model = _language_model()
    tokens = torch.tensor(
        [[1, 4, 7, 3, 2], [2, 5, 8, 6, 1]],
        dtype=torch.long,
    )

    logits = model(tokens)
    loss = model(tokens, labels=tokens.to(torch.int32))
    loss.backward()

    assert logits.shape == (2, 5, 19)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None
    assert torch.count_nonzero(model.token_embedding.weight.grad) > 0
    geometry_gradients = [
        module.geometry.grad for module in _random_feature_modules(model)
    ]
    assert all(gradient is not None for gradient in geometry_gradients)
    assert any(
        gradient is not None and torch.count_nonzero(gradient) > 0
        for gradient in geometry_gradients
    )

    prompt = tokens[:, :2]
    assert model.training
    generated = model.generate(prompt, max_new_tokens=3, top_k=1)

    assert model.training
    assert generated.shape == (2, 5)
    assert torch.equal(generated[:, :2], prompt)
    assert torch.all((generated >= 0) & (generated < model.vocab_size))


def test_encoder_decoder_forward_loss_and_generation() -> None:
    """The encoder-decoder trains and generates with padding masks."""
    torch.manual_seed(31)
    model = _encoder_decoder()
    source_tokens = torch.tensor(
        [[1, 3, 5, 0], [2, 4, 6, 7]],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [[1, 8, 4, 0], [1, 3, 9, 6]],
        dtype=torch.long,
    )
    source_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
    )
    target_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
    )
    labels = target_tokens.masked_fill(~target_mask, -100)

    logits = model(
        source_tokens,
        target_tokens,
        source_mask=source_mask,
        target_mask=target_mask,
    )
    loss = model(
        source_tokens,
        target_tokens,
        source_mask=source_mask,
        target_mask=target_mask,
        labels=labels,
    )
    loss.backward()

    assert logits.shape == (2, 4, 19)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.source_embedding.weight.grad is not None
    assert torch.count_nonzero(model.source_embedding.weight.grad) > 0
    assert model.target_embedding.weight.grad is not None
    assert torch.count_nonzero(model.target_embedding.weight.grad) > 0

    prompt = target_tokens[:, :1]
    assert model.training
    generated = model.generate(
        source_tokens,
        prompt,
        max_new_tokens=2,
        source_mask=source_mask,
        top_k=1,
    )

    assert model.training
    assert generated.shape == (2, 3)
    assert torch.equal(generated[:, :1], prompt)
    assert torch.all((generated >= 0) & (generated < model.target_vocab_size))


def test_language_model_recurrent_logits_match_full_prefix() -> None:
    model = _language_model(depth=2, rotary=True)
    model.eval()
    tokens = torch.tensor(
        [[1, 4, 7, 3, 2], [2, 5, 8, 6, 1]],
        dtype=torch.long,
    )

    with torch.no_grad():
        expected = model(tokens)
        first, state = model.forward_with_state(tokens[:, :3])
        second, state = model.forward_with_state(tokens[:, 3:4], state=state)
        third, state = model.forward_with_state(tokens[:, 4:], state=state)

    actual = torch.cat((first, second, third), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert state.sequence_length == tokens.shape[1]
    assert all(layer.cross_attention is None for layer in state.layers)


def test_encoder_decoder_recurrent_logits_match_full_decode() -> None:
    model = _encoder_decoder(depth=2, rotary=True)
    model.eval()
    source_tokens = torch.tensor(
        [[1, 3, 5, 0], [2, 4, 6, 7]],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [[1, 8, 4, 3], [1, 3, 9, 6]],
        dtype=torch.long,
    )
    source_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
    )

    with torch.no_grad():
        context = model.encode(source_tokens, source_mask=source_mask)
        expected = model.decode(
            target_tokens,
            context,
            source_mask=source_mask,
        )
        first, state = model.decode_with_state(
            target_tokens[:, :2],
            context,
            source_mask=source_mask,
        )
        second, state = model.decode_with_state(
            target_tokens[:, 2:3],
            state=state,
        )
        third, state = model.decode_with_state(
            target_tokens[:, 3:],
            state=state,
        )

    actual = torch.cat((first, second, third), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert state.sequence_length == target_tokens.shape[1]
    assert all(layer.cross_attention is not None for layer in state.layers)


def test_cached_language_generation_uses_chunk_projections() -> None:
    model = _language_model(rotary=True)
    prompt = torch.tensor([[1, 4], [2, 5]], dtype=torch.long)
    expected = _full_prefix_language_generation(model, prompt, 3)
    layer = cast(DarkformerBlock, model.transformer.layers[0])

    with mock.patch.object(
        layer.self_attention.to_qkv,
        "forward",
        wraps=layer.self_attention.to_qkv.forward,
    ) as projection:
        actual = model.generate(prompt, max_new_tokens=3, top_k=1)

    assert torch.equal(actual, expected)
    projected_lengths = [call.args[0].shape[1] for call in projection.call_args_list]
    assert projected_lengths == [2, 1, 1]


def test_cached_encdec_generation_projects_context_once() -> None:
    model = _encoder_decoder(depth=2, rotary=True)
    source_tokens = torch.tensor(
        [[1, 3, 5, 0], [2, 4, 6, 7]],
        dtype=torch.long,
    )
    source_mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
    )
    prompt = torch.tensor([[1], [1]], dtype=torch.long)
    expected = _full_prefix_encoder_decoder_generation(
        model,
        source_tokens,
        prompt,
        3,
        source_mask,
    )
    layers = [cast(DarkformerBlock, module) for module in model.decoder.layers]
    projections = []
    patches = []
    for layer in layers:
        if layer.cross_attention is None:
            raise AssertionError("decoder layer lacks cross-attention")
        patcher = mock.patch.object(
            layer.cross_attention.to_key_value,
            "forward",
            wraps=layer.cross_attention.to_key_value.forward,
        )
        patches.append(patcher)
        projections.append(patcher.start())
    try:
        actual = model.generate(
            source_tokens,
            prompt,
            max_new_tokens=3,
            source_mask=source_mask,
            top_k=1,
        )
    finally:
        for patcher in patches:
            patcher.stop()

    assert torch.equal(actual, expected)
    assert all(projection.call_count == 1 for projection in projections)


def test_generation_falls_back_for_exact_attention() -> None:
    model = _language_model(attention_mode="exact", rotary=True)
    prompt = torch.tensor([[1, 4]], dtype=torch.long)

    with mock.patch.object(
        model,
        "forward_with_state",
        side_effect=AssertionError("recurrent path called"),
    ):
        generated = model.generate(prompt, max_new_tokens=2, top_k=1)

    assert generated.shape == (1, 4)


def test_generation_slices_buffer_after_eos() -> None:
    model = _language_model()
    prompt = torch.tensor([[1, 4], [2, 5]], dtype=torch.long)
    next_token = torch.full((2, 1), 3, dtype=torch.long)

    with mock.patch(
        "darkformer_pytorch.model._sample_next_token", return_value=next_token
    ):
        generated = model.generate(
            prompt,
            max_new_tokens=4,
            eos_token_id=3,
        )

    assert generated.shape == (2, 3)
    assert torch.equal(generated[:, -1], next_token[:, 0])


def test_generation_preflights_prompt_and_length() -> None:
    model = _language_model(max_seq_len=4)

    with pytest.raises(ValueError, match="at least one token"):
        model.generate(
            torch.empty((1, 0), dtype=torch.long),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="requested sequence length"):
        model.generate(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            max_new_tokens=2,
        )


def test_noncausal_language_model_rejects_labels() -> None:
    model = _language_model(causal=False)
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="requires causal attention"):
        model(tokens, labels=tokens.to(torch.int32))


def test_language_model_projection_controls_reach_every_layer() -> None:
    """Language-model projection controls affect the complete stack."""
    model = _language_model(depth=2, deterministic=False)

    _assert_projection_controls(model, expected_modules=2)


def test_encoder_decoder_projection_controls_reach_all_attention() -> None:
    """Encoder-decoder controls include self- and cross-attention."""
    model = _encoder_decoder(deterministic=False)

    _assert_projection_controls(model, expected_modules=3)


def test_model_defaults_use_variance_reduced_feature_estimator() -> None:
    """Model constructors propagate orthogonal features and a zero feature floor."""
    model = _language_model(depth=2)

    for random_features in _random_feature_modules(model):
        assert random_features.orthogonal
        assert random_features.eps == 0.0


def test_fixing_projection_subset_cannot_collide_redraw_seed_streams() -> None:
    """Unequal redraw counters do not alias another layer's projection seed."""
    model = Darkformer(
        dim=4,
        depth=2,
        heads=1,
        head_dim=4,
        num_features=8,
        rotary=False,
        projection_seed=7,
    )
    attention_modules = list(model._attention_modules())
    first = attention_modules[0].attention
    second = attention_modules[1].attention
    second.fix_projection_matrices_()

    model.redraw_projection_matrices_()
    model.redraw_projection_matrices_()

    assert first._redraw_count.item() == 2
    assert second._redraw_count.item() == 0
    assert not torch.equal(
        first.random_features.projection_matrix,
        second.random_features.projection_matrix,
    )


def test_public_models_default_to_256_token_causal_chunks() -> None:
    """Every public model propagates the causal chunk default to its attention."""
    models = (
        Darkformer(dim=4, depth=1, heads=1, rotary=False),
        DarkformerLM(vocab_size=5, dim=4, depth=1, heads=1, rotary=False),
        DarkformerEncDec(
            source_vocab_size=5,
            target_vocab_size=5,
            dim=4,
            depth=1,
            heads=1,
            rotary=False,
        ),
    )

    for model in models:
        attention_modules = [
            module
            for module in model.modules()
            if isinstance(module, DarkformerKernelAttention)
        ]
        assert attention_modules
        assert all(module.causal_chunk_size == 256 for module in attention_modules)


def test_model_propagates_independent_determinism_policies() -> None:
    """High-level constructors preserve separate projection/backend settings."""
    model = _language_model(
        depth=2,
        deterministic=False,
        fixed_projection=True,
        backend_deterministic=False,
    )

    for attention in model.transformer._attention_modules():
        assert attention.attention.random_features.projection_is_fixed
        assert not attention.attention.backend_deterministic


def test_language_model_whitening_calibrates_all_layers() -> None:
    """Representative tokens initialize every language-model geometry."""
    model = _language_model(depth=2, max_seq_len=8)
    tokens = torch.randint(0, model.vocab_size, (4, 8))
    modules = _random_feature_modules(model)
    initial = [module.geometry.detach().clone() for module in modules]

    with pytest.warns(UserWarning, match="geometry_scale=0.5"):
        model.initialize_whitening_(
            tokens,
            regularization=1e-3,
            geometry_scale=0.5,
        )

    assert model.training
    for module, original in zip(modules, initial, strict=True):
        assert not torch.equal(module.geometry, original)


def test_language_model_calibrates_variance_optimal_proposals() -> None:
    """Representative tokens initialize every language-model proposal."""
    model = _language_model(depth=2, max_seq_len=8)
    tokens = torch.randint(0, model.vocab_size, (4, 8))
    modules = _random_feature_modules(model)
    with torch.no_grad():
        for module in modules:
            module.geometry.mul_(0.1)

    model.initialize_variance_optimal_proposal_(tokens)

    assert model.training
    assert all(module.proposal_is_active for module in modules)

    returned = model.reset_variance_optimal_proposal_()

    assert returned is model
    assert all(not module.proposal_is_active for module in modules)


def test_encoder_decoder_calibrates_variance_optimal_proposals() -> None:
    """Representative source and target tokens initialize all proposals."""
    model = _encoder_decoder()
    source_tokens = torch.randint(0, model.source_vocab_size, (4, 8))
    target_tokens = torch.randint(0, model.target_vocab_size, (4, 8))
    modules = _random_feature_modules(model)
    with torch.no_grad():
        for module in modules:
            module.geometry.mul_(0.1)

    model.initialize_variance_optimal_proposal_(source_tokens, target_tokens)

    assert model.training
    assert all(module.proposal_is_active for module in modules)

    returned = model.reset_variance_optimal_proposal_()

    assert returned is model
    assert all(not module.proposal_is_active for module in modules)


def test_encoder_decoder_whitening_uses_normalized_context() -> None:
    """Decoder calibration receives the same context used for inference."""
    model = _encoder_decoder()
    source_tokens = torch.randint(0, model.source_vocab_size, (4, 8))
    target_tokens = torch.randint(0, model.target_vocab_size, (4, 8))

    with (
        mock.patch.object(
            model.encoder,
            "initialize_whitening_",
            wraps=model.encoder.initialize_whitening_,
        ) as encoder_initializer,
        mock.patch.object(
            model.decoder,
            "initialize_whitening_",
            wraps=model.decoder.initialize_whitening_,
        ) as decoder_initializer,
        pytest.warns(UserWarning, match="geometry_scale=0.5"),
    ):
        model.initialize_whitening_(
            source_tokens,
            target_tokens,
            geometry_scale=0.5,
        )

    assert encoder_initializer.call_args.kwargs["geometry_scale"] == 0.5
    assert decoder_initializer.call_args.kwargs["geometry_scale"] == 0.5
    context = decoder_initializer.call_args.kwargs["context"]
    expected = model.encode(source_tokens)
    unnormalized = model.encoder(model.source_embedding(source_tokens))
    torch.testing.assert_close(context, expected)
    assert not torch.allclose(context, unnormalized)


@pytest.mark.parametrize("training", [True, False])
def test_encoder_decoder_whitening_restores_mode_after_error(training: bool) -> None:
    """Calibration restores the original mode when a later stage fails."""
    model = _encoder_decoder()
    model.train(training)
    source_tokens = torch.randint(0, model.source_vocab_size, (4, 8))
    target_tokens = torch.randint(0, model.target_vocab_size, (4, 8))

    with (
        mock.patch.object(
            model.decoder,
            "initialize_whitening_",
            side_effect=RuntimeError("calibration failed"),
        ),
        pytest.raises(RuntimeError, match="calibration failed"),
    ):
        model.initialize_whitening_(source_tokens, target_tokens)

    assert model.training is training


def test_encoder_decoder_whitening_is_deterministic_with_dropout() -> None:
    """Calibration disables dropout for every encoder-decoder stage."""
    model = _encoder_decoder(dropout=0.75)
    model.train()
    source_tokens = torch.randint(0, model.source_vocab_size, (4, 8))
    target_tokens = torch.randint(0, model.target_vocab_size, (4, 8))

    model.initialize_whitening_(source_tokens, target_tokens)
    first = [
        module.geometry.detach().clone() for module in _random_feature_modules(model)
    ]
    torch.randn(128)
    model.initialize_whitening_(source_tokens, target_tokens)
    second = [
        module.geometry.detach().clone() for module in _random_feature_modules(model)
    ]

    assert model.training
    for before, after in zip(first, second, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
