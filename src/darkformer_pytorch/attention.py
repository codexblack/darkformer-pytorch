"""DARKformer attention modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from darkformer_pytorch.backends import AttentionBackend, exact_attention
from darkformer_pytorch.random_features import DataAwareRandomFeatures

AttentionMode = Literal["linear", "auto", "exact"]


@dataclass(slots=True)
class CausalAttentionState:
    """Recurrent state for causal linear attention."""

    key_sum: torch.Tensor
    key_value_sum: torch.Tensor
    key_log_scale: torch.Tensor
    sequence_length: int
    projection_version: int


@dataclass(slots=True)
class ContextAttentionState:
    """Precomputed state for noncausal linear attention."""

    key_sum: torch.Tensor
    key_value_sum: torch.Tensor
    context_length: int
    projection_version: int


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


def _normalize_attention(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    valid = denominator > 0.0
    output = numerator / torch.where(valid, denominator, 1.0)[..., None]
    return torch.where(valid[..., None], output, 0.0)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for query and key tensors."""

    cosine_cache: torch.Tensor
    frequency_indices: torch.Tensor
    sine_cache: torch.Tensor
    _cache_dtype: torch.dtype | None

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        rotary_dim = head_dim - head_dim % 2
        if rotary_dim < 2:
            raise ValueError("rotary embeddings require head_dim >= 2")
        if base <= 0.0 or not math.isfinite(base):
            raise ValueError("rotary base must be finite and positive")
        frequency_indices = torch.arange(0, rotary_dim, 2, dtype=torch.long)
        self.base = float(base)
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self._cache_dtype = None
        self.register_buffer(
            "frequency_indices",
            frequency_indices,
            persistent=False,
        )
        self.register_buffer(
            "cosine_cache",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "sine_cache",
            torch.empty(0),
            persistent=False,
        )

    def _rotary_values(
        self,
        tensor: torch.Tensor,
        offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if offset < 0:
            raise ValueError("rotary offset must be nonnegative")
        required_length = offset + tensor.shape[-2]
        cache_is_valid = (
            self.cosine_cache.device == tensor.device
            and self.cosine_cache.dtype == tensor.dtype
            and self._cache_dtype == tensor.dtype
            and self.cosine_cache.shape[0] >= required_length
        )
        if not cache_is_valid:
            current_length = (
                self.cosine_cache.shape[0]
                if self.cosine_cache.device == tensor.device
                and self.cosine_cache.dtype == tensor.dtype
                and self._cache_dtype == tensor.dtype
                else 0
            )
            cache_length = max(required_length, max(16, 2 * current_length))
            positions = torch.arange(
                cache_length,
                device=tensor.device,
                dtype=torch.float32,
            )
            inverse_frequency = self.base ** (
                -self.frequency_indices.to(
                    device=tensor.device,
                    dtype=torch.float32,
                )
                / self.rotary_dim
            )
            angles = torch.outer(positions, inverse_frequency)
            self.cosine_cache = angles.cos().to(tensor.dtype)
            self.sine_cache = angles.sin().to(tensor.dtype)
            self._cache_dtype = tensor.dtype
        cosine = self.cosine_cache[offset:required_length][None, None, :, :]
        sine = self.sine_cache[offset:required_length][None, None, :, :]
        return cosine, sine

    def _apply_with_values(
        self,
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
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
        *,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the same rotary basis to queries and keys."""
        if query.shape != key.shape:
            raise ValueError(
                "rotary self-attention requires matching query and key shapes"
            )
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"query and key head dimensions must be {self.head_dim}")
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("query and key must share a device and dtype")
        cosine, sine = self._rotary_values(query, offset)
        return (
            self._apply_with_values(query, cosine, sine),
            self._apply_with_values(key, cosine, sine),
        )


class DarkformerKernelAttention(nn.Module):
    """Data-aware attention over projected queries, keys, and values."""

    _calls_since_redraw: torch.Tensor
    _redraw_count: torch.Tensor
    _redraw_seed: torch.Tensor
    _projection_version: int
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
        self._projection_version = 0
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
        redraw_seed = (
            torch.randint(0, 2**31 - 1, (), dtype=torch.long)
            if projection_seed is None
            else torch.tensor(projection_seed % (2**63 - 1), dtype=torch.long)
        )
        self.register_buffer("_redraw_seed", redraw_seed, persistent=True)

    @property
    def geometry(self) -> nn.Parameter:
        """Learned factor of the attention covariance."""
        return self.random_features.geometry

    def covariance(self) -> torch.Tensor:
        """Return the attention covariance for every head."""
        return self.random_features.covariance()

    def _redraw_generator(self) -> torch.Generator:
        device = self.random_features.projection_matrix.device
        generator = torch.Generator(device=device)
        seed = int(self._redraw_seed.item()) + int(self._redraw_count.item()) + 1
        generator.manual_seed(seed % (2**63 - 1))
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
            self._projection_version += 1
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
        transformed_query = torch.matmul(
            query * normalizer,
            geometry.transpose(-1, -2),
        )
        transformed_key = torch.matmul(
            key * normalizer,
            geometry.transpose(-1, -2),
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
        )
        key_value = torch.matmul(key_features.transpose(-1, -2), value)
        numerator = torch.matmul(query_features, key_value)
        return _normalize_attention(numerator, denominator)

    def _validate_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> None:
        for name, tensor in (("key", key), ("value", value)):
            if tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [batch, heads, length, dim]")
        if key.shape[:3] != value.shape[:3]:
            raise ValueError("key and value batch, head, and length dimensions differ")
        if key.shape[1] != self.num_heads:
            raise ValueError(f"key must have {self.num_heads} heads")
        if key.shape[-1] != self.head_dim:
            raise ValueError(f"key head dimension must be {self.head_dim}")
        if key.shape[2] < 1:
            raise ValueError("key length must be positive")
        if key.device != value.device:
            raise ValueError("key and value must share a device")
        if key.dtype != value.dtype:
            raise ValueError("key and value must share a dtype")
        _validate_mask(
            key_mask,
            batch_size=key.shape[0],
            length=key.shape[2],
            device=key.device,
            name="key_mask",
        )

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor | None,
        key_mask: torch.Tensor | None,
    ) -> None:
        if query.ndim != 4:
            raise ValueError("query must have shape [batch, heads, length, dim]")
        self._validate_key_value(key, value, key_mask)
        if query.shape[:2] != key.shape[:2]:
            raise ValueError("query and key batch and head dimensions differ")
        if query.shape[1] != self.num_heads:
            raise ValueError(f"query must have {self.num_heads} heads")
        if query.shape[-1] != self.head_dim:
            raise ValueError(f"query head dimension must be {self.head_dim}")
        if query.shape[2] < 1:
            raise ValueError("query length must be positive")
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("query, key, and value must share a device and dtype")
        _validate_mask(
            query_mask,
            batch_size=query.shape[0],
            length=query.shape[2],
            device=query.device,
            name="query_mask",
        )
        if self.causal and query.shape[2] != key.shape[2]:
            raise ValueError("causal attention requires matching query and key lengths")

    def _causal_attention_with_state(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: torch.Tensor | None,
        key_mask: torch.Tensor | None,
        state: CausalAttentionState | None,
    ) -> tuple[torch.Tensor, CausalAttentionState]:
        query_logits = self.random_features.feature_logits(query)
        key_logits = self.random_features.feature_logits(key)
        query_maximum = query_logits.amax(dim=-1, keepdim=True).detach().clamp_min(0.0)
        feature_scale = self.num_features**-0.5
        query_features = torch.exp(query_logits - query_maximum)
        query_features = query_features + self.eps * torch.exp(-query_maximum)
        query_features = query_features * feature_scale
        value_dtype = value.dtype
        value = value.to(query_features.dtype)
        batch, heads, _, features = query_features.shape
        value_dim = value.shape[-1]
        if state is None:
            key_state = query_features.new_zeros(batch, heads, features)
            key_value_state = query_features.new_zeros(
                batch,
                heads,
                features,
                value_dim,
            )
            key_log_scale = query_features.new_full(
                (batch, heads, 1, 1),
                0.0,
            )
            sequence_length = 0
        else:
            expected_key_shape = (batch, heads, features)
            expected_value_shape = (batch, heads, features, value_dim)
            expected_scale_shape = (batch, heads, 1, 1)
            if tuple(state.key_sum.shape) != expected_key_shape:
                raise ValueError(f"state.key_sum must have shape {expected_key_shape}")
            if tuple(state.key_value_sum.shape) != expected_value_shape:
                raise ValueError(
                    f"state.key_value_sum must have shape {expected_value_shape}"
                )
            if tuple(state.key_log_scale.shape) != expected_scale_shape:
                raise ValueError(
                    f"state.key_log_scale must have shape {expected_scale_shape}"
                )
            for name, tensor in (
                ("state.key_sum", state.key_sum),
                ("state.key_value_sum", state.key_value_sum),
                ("state.key_log_scale", state.key_log_scale),
            ):
                if tensor.device != query.device:
                    raise ValueError(f"{name} must be on the same device as query")
                if tensor.dtype != query_features.dtype:
                    raise ValueError(f"{name} must have dtype {query_features.dtype}")
            if state.sequence_length < 0:
                raise ValueError("state.sequence_length must be nonnegative")
            if state.projection_version != self._projection_version:
                raise RuntimeError(
                    "projection matrices changed after the state was created"
                )
            key_state = state.key_sum
            key_value_state = state.key_value_sum
            key_log_scale = state.key_log_scale
            sequence_length = state.sequence_length
        outputs = []
        start = 0
        while start < query_features.shape[2]:
            window_stop = min(
                start + self.causal_chunk_size,
                query_features.shape[2],
            )
            window_logits = key_logits[:, :, start:window_stop]
            window_mask = None if key_mask is None else key_mask[:, start:window_stop]
            if window_mask is not None:
                window_logits = window_logits.masked_fill(
                    ~window_mask[:, None, :, None],
                    -torch.inf,
                )
            token_scale = window_logits.amax(dim=-1, keepdim=True).detach()
            first_scale = torch.where(
                torch.isfinite(token_scale[:, :, :1]),
                token_scale[:, :, :1],
                0.0,
            ).clamp_min(0.0)
            calculation_scale = torch.maximum(key_log_scale, first_scale)
            excessive_rise = token_scale > calculation_scale + 16.0
            excessive_rise[:, :, 0] = False
            rise_positions = excessive_rise.any(dim=(0, 1, 3)).nonzero()
            if rise_positions.numel() > 0:
                window_stop = start + int(rise_positions[0, 0].item())

            stop = window_stop
            query_chunk = query_features[:, :, start:stop]
            key_logits_chunk = key_logits[:, :, start:stop]
            value_chunk = value[:, :, start:stop]
            key_mask_chunk = None if key_mask is None else key_mask[:, start:stop]
            previous_factor = torch.exp(key_log_scale - calculation_scale)
            key_state = key_state * previous_factor.squeeze(-1)
            key_value_state = key_value_state * previous_factor
            scaled_key_logits = key_logits_chunk - calculation_scale
            if key_mask_chunk is not None:
                scaled_key_logits = scaled_key_logits.masked_fill(
                    ~key_mask_chunk[:, None, :, None],
                    -torch.inf,
                )
            key_chunk = torch.exp(scaled_key_logits)
            key_chunk = key_chunk + self.eps * torch.exp(-calculation_scale)
            key_chunk = key_chunk * feature_scale
            if key_mask_chunk is not None:
                key_chunk = key_chunk.masked_fill(
                    ~key_mask_chunk[:, None, :, None],
                    0.0,
                )
            prefix_scores = torch.matmul(
                query_chunk,
                key_chunk.transpose(-1, -2),
            ).tril_()
            denominator = torch.einsum(
                "bhnm,bhm->bhn",
                query_chunk,
                key_state,
            )
            denominator = denominator + prefix_scores.sum(dim=-1)
            numerator = torch.matmul(query_chunk, key_value_state)
            numerator = numerator + torch.matmul(prefix_scores, value_chunk)
            outputs.append(_normalize_attention(numerator, denominator))
            key_state = key_state + key_chunk.sum(dim=2)
            key_value_state = key_value_state + torch.matmul(
                key_chunk.transpose(-1, -2),
                value_chunk,
            )
            if key_mask_chunk is not None:
                key_logits_chunk = key_logits_chunk.masked_fill(
                    ~key_mask_chunk[:, None, :, None],
                    -torch.inf,
                )
            chunk_scale = key_logits_chunk.amax(
                dim=(-2, -1),
                keepdim=True,
            ).detach()
            chunk_scale = torch.where(
                torch.isfinite(chunk_scale),
                chunk_scale,
                0.0,
            ).clamp_min(0.0)
            next_scale = torch.maximum(calculation_scale, chunk_scale)
            next_factor = torch.exp(calculation_scale - next_scale)
            key_state = key_state * next_factor.squeeze(-1)
            key_value_state = key_value_state * next_factor
            key_log_scale = next_scale
            start = stop
        output = torch.cat(outputs, dim=2)
        if query_mask is not None:
            output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
        next_state = CausalAttentionState(
            key_sum=key_state,
            key_value_sum=key_value_state,
            key_log_scale=key_log_scale,
            sequence_length=sequence_length + query.shape[2],
            projection_version=self._projection_version,
        )
        return output.to(value_dtype), next_state

    def forward_with_state(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        state: CausalAttentionState | None = None,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CausalAttentionState]:
        """Apply causal linear attention and return its recurrent state."""
        if not self.causal or self.attention_mode != "linear":
            raise RuntimeError(
                "recurrent state requires causal attention_mode='linear'"
            )
        self._validate_inputs(query, key, value, query_mask, key_mask)
        if state is None:
            self._maybe_redraw()
        return self._causal_attention_with_state(
            query,
            key,
            value,
            query_mask=query_mask,
            key_mask=key_mask,
            state=state,
        )

    def build_context_state(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
    ) -> ContextAttentionState:
        """Precompute the sufficient statistics for linear cross-attention."""
        if self.causal or self.attention_mode != "linear":
            raise RuntimeError(
                "context state requires noncausal attention_mode='linear'"
            )
        self._validate_key_value(key, value, key_mask)
        self._maybe_redraw()
        key_features = self.random_features.key_feature_map(
            key,
            key_mask=key_mask,
        )
        value = value.to(key_features.dtype)
        return ContextAttentionState(
            key_sum=key_features.sum(dim=-2),
            key_value_sum=torch.matmul(key_features.transpose(-1, -2), value),
            context_length=key.shape[2],
            projection_version=self._projection_version,
        )

    def forward_with_context_state(
        self,
        query: torch.Tensor,
        state: ContextAttentionState,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply linear attention against precomputed context statistics."""
        if self.causal or self.attention_mode != "linear":
            raise RuntimeError(
                "context state requires noncausal attention_mode='linear'"
            )
        if query.ndim != 4:
            raise ValueError("query must have shape [batch, heads, length, dim]")
        if query.shape[1] != self.num_heads or query.shape[-1] != self.head_dim:
            raise ValueError(
                f"query must have {self.num_heads} heads and head_dim {self.head_dim}"
            )
        if query.shape[2] < 1:
            raise ValueError("query length must be positive")
        _validate_mask(
            query_mask,
            batch_size=query.shape[0],
            length=query.shape[2],
            device=query.device,
            name="query_mask",
        )
        if state.context_length < 1:
            raise ValueError("state.context_length must be positive")
        if state.projection_version != self._projection_version:
            raise RuntimeError(
                "projection matrices changed after the state was created"
            )
        expected_key_shape = (query.shape[0], self.num_heads, self.num_features)
        if tuple(state.key_sum.shape) != expected_key_shape:
            raise ValueError(f"state.key_sum must have shape {expected_key_shape}")
        if (
            state.key_value_sum.ndim != 4
            or tuple(state.key_value_sum.shape[:3]) != expected_key_shape
        ):
            raise ValueError(
                "state.key_value_sum must have shape "
                f"{(*expected_key_shape, 'value_dim')}"
            )
        query_features = self.random_features.query_feature_map(query)
        for name, tensor in (
            ("state.key_sum", state.key_sum),
            ("state.key_value_sum", state.key_value_sum),
        ):
            if tensor.device != query.device:
                raise ValueError(f"{name} must be on the same device as query")
            if tensor.dtype != query_features.dtype:
                raise ValueError(f"{name} must have dtype {query_features.dtype}")
        denominator = torch.einsum(
            "bhnm,bhm->bhn",
            query_features,
            state.key_sum,
        )
        numerator = torch.matmul(query_features, state.key_value_sum)
        output = _normalize_attention(numerator, denominator)
        if query_mask is not None:
            output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
        return output.to(query.dtype)

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
        self._validate_inputs(query, key, value, query_mask, key_mask)
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

        self._maybe_redraw()
        if self.causal:
            output, _ = self._causal_attention_with_state(
                query,
                key,
                value,
                query_mask=query_mask,
                key_mask=key_mask,
                state=None,
            )
            return output
        query_features, key_features = self.random_features(
            query,
            key,
            key_mask=key_mask,
        )
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

    def _project_inputs(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [batch, length, {self.dim}]")
        batch, length, _ = inputs.shape
        return tuple(
            self.to_qkv(inputs)
            .reshape(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

    def _project_output(
        self,
        output: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, length, _ = output.shape
        output = output.transpose(1, 2).reshape(batch, length, self.inner_dim)
        output = self.output_dropout(self.to_output(output))
        if mask is not None:
            output = output.masked_fill(~mask[:, :, None], 0.0)
        return output

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention to a batch-first sequence."""
        query, key, value = self._project_inputs(inputs)
        if self.rotary is not None:
            query, key = self.rotary(query, key)
        output = self.attention(
            query,
            key,
            value,
            query_mask=mask,
            key_mask=mask,
        )
        return self._project_output(output, mask)

    def forward_with_state(
        self,
        inputs: torch.Tensor,
        *,
        state: CausalAttentionState | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CausalAttentionState]:
        """Apply causal self-attention and return its recurrent state."""
        query, key, value = self._project_inputs(inputs)
        if self.rotary is not None:
            offset = 0 if state is None else state.sequence_length
            query, key = self.rotary(query, key, offset=offset)
        output, next_state = self.attention.forward_with_state(
            query,
            key,
            value,
            state=state,
            query_mask=mask,
            key_mask=mask,
        )
        return self._project_output(output, mask), next_state


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
        if min(dim, context_dim, heads) < 1:
            raise ValueError("dimensions and heads must be positive")
        if head_dim is None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when head_dim is omitted")
        head_dim = dim // heads if head_dim is None else head_dim
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
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

    def _project_query(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [batch, length, {self.dim}]")
        batch, length, _ = inputs.shape
        return (
            self.to_query(inputs)
            .reshape(
                batch,
                length,
                self.heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def _project_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [batch, context_length, {self.context_dim}]"
            )
        batch, length, _ = context.shape
        return tuple(
            self.to_key_value(context)
            .reshape(batch, length, 2, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

    def _project_output(
        self,
        output: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, length, _ = output.shape
        output = output.transpose(1, 2).reshape(batch, length, self.inner_dim)
        output = self.output_dropout(self.to_output(output))
        if mask is not None:
            output = output.masked_fill(~mask[:, :, None], 0.0)
        return output

    def build_context_state(
        self,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> ContextAttentionState:
        """Project and summarize context for repeated cross-attention."""
        key, value = self._project_context(context)
        return self.attention.build_context_state(
            key,
            value,
            key_mask=context_mask,
        )

    def forward_with_state(
        self,
        inputs: torch.Tensor,
        state: ContextAttentionState,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to a previously summarized context."""
        query = self._project_query(inputs)
        output = self.attention.forward_with_context_state(
            query,
            state,
            query_mask=mask,
        )
        return self._project_output(output, mask)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from a batch-first sequence to a context sequence."""
        query = self._project_query(inputs)
        key, value = self._project_context(context)
        if inputs.shape[0] != context.shape[0]:
            raise ValueError("inputs and context batch sizes must match")
        output = self.attention(
            query,
            key,
            value,
            query_mask=mask,
            key_mask=context_mask,
        )
        return self._project_output(output, mask)


DarkformerAttention = SelfAttention


__all__ = [
    "AttentionMode",
    "CausalAttentionState",
    "ContextAttentionState",
    "CrossAttention",
    "DarkformerAttention",
    "DarkformerKernelAttention",
    "RotaryEmbedding",
    "SelfAttention",
]
