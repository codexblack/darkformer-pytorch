"""DARKformer attention modules."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from darkformer.backends import AttentionBackend, exact_attention
from darkformer.random_features import DataAwareRandomFeatures

AttentionMode = Literal["linear", "auto", "exact"]


def _feature_count(head_dim: int) -> int:
    return max(1, int(head_dim * math.log(max(head_dim, 2))))


def _validate_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
    name: str,
) -> None:
    if mask is None:
        return
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != device:
        raise ValueError(f"{name} must be on the same device as its input")
    if tuple(mask.shape) != (batch_size, length):
        raise ValueError(f"{name} must have shape {(batch_size, length)}")


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for query and key tensors."""

    inverse_frequency: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        rotary_dim = head_dim - head_dim % 2
        if rotary_dim < 2:
            raise ValueError("rotary embeddings require head_dim >= 2")
        if base <= 0.0 or not math.isfinite(base):
            raise ValueError("rotary base must be finite and positive")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    def _apply_rotary(self, tensor: torch.Tensor) -> torch.Tensor:
        length = tensor.shape[-2]
        positions = torch.arange(
            length,
            device=tensor.device,
            dtype=self.inverse_frequency.dtype,
        )
        angles = torch.outer(positions, self.inverse_frequency)
        cosine = angles.cos().to(tensor.dtype)[None, None, :, :]
        sine = angles.sin().to(tensor.dtype)[None, None, :, :]
        rotary = tensor[..., : self.rotary_dim]
        remainder = tensor[..., self.rotary_dim :]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine),
            dim=-1,
        )
        if remainder.shape[-1] == 0:
            return rotated
        return torch.cat((rotated, remainder), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same rotary basis to queries and keys."""
        if query.shape[-2:] != key.shape[-2:]:
            raise ValueError(
                "rotary self-attention requires matching query and key shapes"
            )
        return self._apply_rotary(query), self._apply_rotary(key)


class DarkformerKernelAttention(nn.Module):
    """Data-aware attention over projected queries, keys, and values."""

    _calls_since_redraw: torch.Tensor
    _redraw_count: torch.Tensor
    random_features: DataAwareRandomFeatures

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        *,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        per_head_geometry: bool = True,
        orthogonal_features: bool = True,
        causal: bool = False,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        causal_chunk_size: int = 64,
        eps: float = 1e-6,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        num_features = (
            _feature_count(head_dim) if num_features is None else num_features
        )
        if attention_mode not in ("linear", "auto", "exact"):
            raise ValueError(f"unknown attention mode: {attention_mode!r}")
        if exact_threshold is not None and exact_threshold < 1:
            raise ValueError("exact_threshold must be positive")
        if causal_chunk_size < 1:
            raise ValueError("causal_chunk_size must be positive")
        if feature_redraw_interval is not None and feature_redraw_interval < 1:
            raise ValueError("feature_redraw_interval must be positive")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_features = num_features
        self.causal = causal
        self.attention_mode = attention_mode
        self.exact_threshold = (
            num_features if exact_threshold is None else exact_threshold
        )
        self.exact_backend = exact_backend
        self.causal_chunk_size = causal_chunk_size
        self.eps = float(eps)
        self.feature_redraw_interval = feature_redraw_interval
        self.projection_seed = projection_seed
        self.deterministic = deterministic
        self.random_features = DataAwareRandomFeatures(
            head_dim=head_dim,
            num_heads=num_heads,
            num_features=num_features,
            rank=geometry_rank,
            per_head=per_head_geometry,
            orthogonal=orthogonal_features,
            eps=eps,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        self.register_buffer(
            "_calls_since_redraw",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_redraw_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    @property
    def geometry(self) -> nn.Parameter:
        """Learned factor of the attention covariance."""
        return self.random_features.geometry

    def covariance(self) -> torch.Tensor:
        """Return the attention covariance for every head."""
        return self.random_features.covariance()

    def _redraw_generator(self) -> torch.Generator | None:
        if self.projection_seed is None:
            return None
        device = self.random_features.projection_matrix.device
        generator = torch.Generator(device=device)
        generator.manual_seed(self.projection_seed + int(self._redraw_count.item()) + 1)
        return generator

    @torch.no_grad()
    def redraw_projection_matrices_(
        self,
        *,
        force: bool = False,
    ) -> DarkformerKernelAttention:
        """Redraw the random-feature basis in place."""
        was_fixed = self.random_features.projection_is_fixed
        self.random_features.redraw_projection_(
            generator=self._redraw_generator(),
            force=force,
        )
        if not was_fixed or force:
            self._redraw_count.add_(1)
            self._calls_since_redraw.zero_()
        return self

    @torch.no_grad()
    def fix_projection_matrices_(self) -> DarkformerKernelAttention:
        """Disable automatic and ordinary projection redraws."""
        self.random_features.fix_projection_matrix_()
        return self

    @torch.no_grad()
    def unfix_projection_matrices_(self) -> DarkformerKernelAttention:
        """Enable projection redraws."""
        self.random_features.unfix_projection_matrix_()
        return self

    def _maybe_redraw(self) -> None:
        if (
            not self.training
            or self.feature_redraw_interval is None
            or self.random_features.projection_is_fixed
        ):
            return
        self._calls_since_redraw.add_(1)
        if int(self._calls_since_redraw.item()) >= self.feature_redraw_interval:
            self.redraw_projection_matrices_()

    def _use_exact_attention(self, query_length: int, key_length: int) -> bool:
        if self.attention_mode == "exact":
            return True
        if self.attention_mode == "linear":
            return False
        return max(query_length, key_length) <= self.exact_threshold

    def _transformed_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self.random_features._geometry_for_heads()
        normalizer = self.head_dim**-0.25
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
        return transformed_query, transformed_key

    def _noncausal_attention(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        accumulation_dtype = query_features.dtype
        value = value.to(accumulation_dtype)
        key_sum = key_features.sum(dim=-2)
        denominator = torch.einsum(
            "bhnm,bhm->bhn",
            query_features,
            key_sum,
        ).clamp_min(self.eps)
        key_value = torch.matmul(key_features.transpose(-1, -2), value)
        numerator = torch.matmul(query_features, key_value)
        return numerator / denominator[..., None]

    def _causal_attention(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value = value.to(query_features.dtype)
        batch, heads, _, features = query_features.shape
        value_dim = value.shape[-1]
        key_state = query_features.new_zeros(batch, heads, features)
        key_value_state = query_features.new_zeros(
            batch,
            heads,
            features,
            value_dim,
        )
        outputs = []
        for start in range(0, query_features.shape[2], self.causal_chunk_size):
            stop = min(start + self.causal_chunk_size, query_features.shape[2])
            query_chunk = query_features[:, :, start:stop]
            key_chunk = key_features[:, :, start:stop]
            value_chunk = value[:, :, start:stop]
            key_prefix = key_chunk.cumsum(dim=2) + key_state[:, :, None]
            outer_products = key_chunk[..., None] * value_chunk[..., None, :]
            key_value_prefix = (
                outer_products.cumsum(dim=2) + key_value_state[:, :, None]
            )
            denominator = (query_chunk * key_prefix).sum(dim=-1).clamp_min(self.eps)
            numerator = torch.matmul(
                query_chunk.unsqueeze(-2),
                key_value_prefix,
            ).squeeze(-2)
            outputs.append(numerator / denominator[..., None])
            key_state = key_prefix[:, :, -1]
            key_value_state = key_value_prefix[:, :, -1]
        return torch.cat(outputs, dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply data-aware normalized attention."""
        for name, tensor in (("query", query), ("key", key), ("value", value)):
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [batch, heads, length, dim]"
                )
        if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
            raise ValueError(
                "query, key, and value batch, head, and key lengths differ"
            )
        if query.shape[1] != self.num_heads:
            raise ValueError(f"query must have {self.num_heads} heads")
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError(f"query and key head dimensions must be {self.head_dim}")
        if query.shape[2] < 1 or key.shape[2] < 1:
            raise ValueError("query and key lengths must be positive")
        if query.device != key.device or query.device != value.device:
            raise ValueError("query, key, and value must share a device")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("query, key, and value must share a dtype")
        _validate_mask(
            query_mask,
            batch_size=query.shape[0],
            length=query.shape[2],
            device=query.device,
            name="query_mask",
        )
        _validate_mask(
            key_mask,
            batch_size=key.shape[0],
            length=key.shape[2],
            device=key.device,
            name="key_mask",
        )
        if self.causal and query.shape[2] != key.shape[2]:
            raise ValueError("causal attention requires matching query and key lengths")

        self._maybe_redraw()
        if self._use_exact_attention(query.shape[2], key.shape[2]):
            transformed_query, transformed_key = self._transformed_query_key(
                query,
                key,
            )
            output = exact_attention(
                transformed_query,
                transformed_key,
                value,
                causal=self.causal,
                query_mask=query_mask,
                key_mask=key_mask,
                dropout_p=0.0,
                backend=self.exact_backend,
                scale=1.0,
                deterministic=self.deterministic,
            )
            return output

        query_features, key_features = self.random_features(
            query,
            key,
            key_mask=key_mask,
        )
        if self.causal:
            output = self._causal_attention(query_features, key_features, value)
        else:
            output = self._noncausal_attention(
                query_features,
                key_features,
                value,
            )
        if query_mask is not None:
            output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
        return output.to(value.dtype)


class SelfAttention(nn.Module):
    """Multi-head DARKformer self-attention."""

    attention: DarkformerKernelAttention
    rotary: RotaryEmbedding | None

    def __init__(
        self,
        dim: int,
        *,
        heads: int = 8,
        head_dim: int | None = None,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        per_head_geometry: bool = True,
        orthogonal_features: bool = True,
        causal: bool = False,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        dropout: float = 0.0,
        qkv_bias: bool = False,
        output_bias: bool = True,
        rotary: bool = False,
        rotary_base: float = 10_000.0,
        causal_chunk_size: int = 64,
        eps: float = 1e-6,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1 or heads < 1:
            raise ValueError("dim and heads must be positive")
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        head_dim = dim // heads if head_dim is None else head_dim
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_bias)
        self.rotary = RotaryEmbedding(head_dim, rotary_base) if rotary else None
        self.attention = DarkformerKernelAttention(
            head_dim=head_dim,
            num_heads=heads,
            num_features=num_features,
            geometry_rank=geometry_rank,
            per_head_geometry=per_head_geometry,
            orthogonal_features=orthogonal_features,
            causal=causal,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            causal_chunk_size=causal_chunk_size,
            eps=eps,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        self.to_output = nn.Linear(self.inner_dim, dim, bias=output_bias)
        self.output_dropout = nn.Dropout(dropout)

    @property
    def geometry(self) -> nn.Parameter:
        """Learned factor of the attention covariance."""
        return self.attention.geometry

    def covariance(self) -> torch.Tensor:
        """Return the attention covariance for every head."""
        return self.attention.covariance()

    def redraw_projection_matrices_(self, *, force: bool = False) -> SelfAttention:
        """Redraw the random-feature basis in place."""
        self.attention.redraw_projection_matrices_(force=force)
        return self

    def fix_projection_matrices_(self) -> SelfAttention:
        """Disable projection redraws."""
        self.attention.fix_projection_matrices_()
        return self

    def unfix_projection_matrices_(self) -> SelfAttention:
        """Enable projection redraws."""
        self.attention.unfix_projection_matrices_()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention to a batch-first sequence."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [batch, length, {self.dim}]")
        batch, length, _ = inputs.shape
        query, key, value = (
            self.to_qkv(inputs)
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.rotary is not None:
            query, key = self.rotary(query, key)
        output = self.attention(
            query,
            key,
            value,
            query_mask=mask,
            key_mask=mask,
        )
        output = output.transpose(1, 2).reshape(batch, length, self.inner_dim)
        return self.output_dropout(self.to_output(output))


class CrossAttention(nn.Module):
    """Multi-head DARKformer cross-attention."""

    attention: DarkformerKernelAttention

    def __init__(
        self,
        dim: int,
        *,
        context_dim: int | None = None,
        heads: int = 8,
        head_dim: int | None = None,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        per_head_geometry: bool = True,
        orthogonal_features: bool = True,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        dropout: float = 0.0,
        qkv_bias: bool = False,
        output_bias: bool = True,
        causal_chunk_size: int = 64,
        eps: float = 1e-6,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        context_dim = dim if context_dim is None else context_dim
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        head_dim = dim // heads if head_dim is None else head_dim
        if min(dim, context_dim, heads, head_dim) < 1:
            raise ValueError("dimensions and heads must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.dim = dim
        self.context_dim = context_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_query = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.to_key_value = nn.Linear(
            context_dim,
            2 * self.inner_dim,
            bias=qkv_bias,
        )
        self.attention = DarkformerKernelAttention(
            head_dim=head_dim,
            num_heads=heads,
            num_features=num_features,
            geometry_rank=geometry_rank,
            per_head_geometry=per_head_geometry,
            orthogonal_features=orthogonal_features,
            causal=False,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            causal_chunk_size=causal_chunk_size,
            eps=eps,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        self.to_output = nn.Linear(self.inner_dim, dim, bias=output_bias)
        self.output_dropout = nn.Dropout(dropout)

    @property
    def geometry(self) -> nn.Parameter:
        """Learned factor of the attention covariance."""
        return self.attention.geometry

    def covariance(self) -> torch.Tensor:
        """Return the attention covariance for every head."""
        return self.attention.covariance()

    def redraw_projection_matrices_(self, *, force: bool = False) -> CrossAttention:
        """Redraw the random-feature basis in place."""
        self.attention.redraw_projection_matrices_(force=force)
        return self

    def fix_projection_matrices_(self) -> CrossAttention:
        """Disable projection redraws."""
        self.attention.fix_projection_matrices_()
        return self

    def unfix_projection_matrices_(self) -> CrossAttention:
        """Enable projection redraws."""
        self.attention.unfix_projection_matrices_()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from a batch-first sequence to a context sequence."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [batch, length, {self.dim}]")
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, context_length, {self.context_dim}]"
            )
        if inputs.shape[0] != context.shape[0]:
            raise ValueError("inputs and context batch sizes must match")
        batch, query_length, _ = inputs.shape
        key_length = context.shape[1]
        query = (
            self.to_query(inputs)
            .reshape(batch, query_length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        key, value = (
            self.to_key_value(context)
            .reshape(batch, key_length, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        output = self.attention(
            query,
            key,
            value,
            query_mask=mask,
            key_mask=context_mask,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            query_length,
            self.inner_dim,
        )
        return self.output_dropout(self.to_output(output))


DarkformerAttention = SelfAttention


__all__ = [
    "AttentionMode",
    "CrossAttention",
    "DarkformerAttention",
    "DarkformerKernelAttention",
    "RotaryEmbedding",
    "SelfAttention",
]
