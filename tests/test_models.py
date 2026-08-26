"""Tests for DARKformer language and encoder-decoder models."""

import torch
from torch import nn

from darkformer.model import DarkformerEncDec, DarkformerLM
from darkformer.random_features import DataAwareRandomFeatures

ProjectionModel = DarkformerLM | DarkformerEncDec


def _language_model(
    *,
    depth: int = 1,
    deterministic: bool = True,
) -> DarkformerLM:
    return DarkformerLM(
        vocab_size=19,
        dim=8,
        depth=depth,
        heads=2,
        head_dim=4,
        num_features=8,
        mlp_dim=16,
        max_seq_len=10,
        causal=True,
        attention_mode="linear",
        dropout=0.0,
        rotary=False,
        causal_chunk_size=2,
        projection_seed=7,
        deterministic=deterministic,
    )


def _encoder_decoder(
    *,
    deterministic: bool = True,
) -> DarkformerEncDec:
    return DarkformerEncDec(
        source_vocab_size=17,
        target_vocab_size=19,
        dim=8,
        depth=1,
        heads=2,
        head_dim=4,
        num_features=8,
        mlp_dim=16,
        max_source_length=8,
        max_target_length=8,
        attention_mode="linear",
        dropout=0.0,
        rotary=False,
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


def test_language_model_logits_loss_and_generation() -> None:
    """The language model trains and extends prompts on CPU."""
    torch.manual_seed(29)
    model = _language_model()
    tokens = torch.tensor(
        [[1, 4, 7, 3, 2], [2, 5, 8, 6, 1]],
        dtype=torch.long,
    )

    logits = model(tokens)
    loss = model(tokens, labels=tokens)
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


def test_language_model_projection_controls_reach_every_layer() -> None:
    """Language-model projection controls affect the complete stack."""
    model = _language_model(depth=2, deterministic=False)

    _assert_projection_controls(model, expected_modules=2)


def test_encoder_decoder_projection_controls_reach_all_attention() -> None:
    """Encoder-decoder controls include self- and cross-attention."""
    model = _encoder_decoder(deterministic=False)

    _assert_projection_controls(model, expected_modules=3)
