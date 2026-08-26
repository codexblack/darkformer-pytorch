"""Tests for DARKformer attention modules."""

import pytest
import torch

import darkformer.attention as attention_module
from darkformer.attention import (
    CrossAttention,
    DarkformerKernelAttention,
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
        output_bias=False,
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
        output_bias=False,
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
