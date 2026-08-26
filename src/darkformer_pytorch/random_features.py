"""Data-aware positive random features."""

from __future__ import annotations

import math

import torch
from torch import nn


def _gaussian_projection(
    rows: int,
    columns: int,
    *,
    orthogonal: bool,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw a Gaussian or Gaussian-orthogonal projection matrix."""
    if not orthogonal:
        return torch.randn(
            rows,
            columns,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

    blocks = []
    remaining = rows
    while remaining > 0:
        unstructured = torch.randn(
            columns,
            columns,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        orthogonal_block, _ = torch.linalg.qr(unstructured, mode="reduced")
        block_rows = min(remaining, columns)
        blocks.append(orthogonal_block.transpose(0, 1)[:block_rows])
        remaining -= block_rows

    projection = torch.cat(blocks, dim=0)
    gaussian = torch.randn(
        rows,
        columns,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    row_norms = torch.linalg.vector_norm(gaussian, dim=1)
    return projection * row_norms[:, None]


class DataAwareRandomFeatures(nn.Module):
    """Positive random features with a learned Mahalanobis geometry.

    The learned matrix ``M`` is stored in ``geometry``. The covariance is
    ``M.transpose(-1, -2) @ M`` and is never materialized during attention.

    Args:
      head_dim: Query and key size for one attention head.
      num_heads: Number of attention heads.
      num_features: Number of positive random features.
      rank: Rank of the learned geometry. Defaults to ``head_dim``.
      per_head: Whether each head has an independent geometry.
      orthogonal: Whether to use Gaussian-orthogonal base projections.
      eps: Positive offset added to feature values.
      projection_seed: Optional seed for projection initialization.
      deterministic: Whether to start with a fixed projection matrix.
    """

    geometry: nn.Parameter
    projection_matrix: torch.Tensor
    _projection_fixed: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_features: int,
        rank: int | None = None,
        per_head: bool = True,
        orthogonal: bool = False,
        eps: float = 1e-6,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        rank = head_dim if rank is None else rank
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if num_features < 1:
            raise ValueError("num_features must be positive")
        if not 1 <= rank <= head_dim:
            raise ValueError("rank must be in the interval [1, head_dim]")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_features = num_features
        self.rank = rank
        self.per_head = per_head
        self.orthogonal = orthogonal
        self.eps = float(eps)
        self.projection_seed = projection_seed

        geometry_heads = num_heads if per_head else 1
        geometry = torch.empty(geometry_heads, rank, head_dim)
        if rank == head_dim:
            geometry.copy_(torch.eye(head_dim).expand(geometry_heads, -1, -1))
        else:
            for head_geometry in geometry:
                nn.init.orthogonal_(head_geometry)
        self.geometry = nn.Parameter(geometry)

        generator = None
        if projection_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(projection_seed)
        projection = _gaussian_projection(
            num_features,
            rank,
            orthogonal=orthogonal,
            generator=generator,
        )
        self.register_buffer("projection_matrix", projection, persistent=True)
        self.register_buffer(
            "_projection_fixed",
            torch.tensor(bool(deterministic), dtype=torch.bool),
            persistent=True,
        )

    @property
    def projection_is_fixed(self) -> bool:
        """Whether ordinary redraw requests are disabled."""
        return bool(self._projection_fixed.item())

    @torch.no_grad()
    def redraw_projection_(
        self,
        generator: torch.Generator | None = None,
        *,
        force: bool = False,
    ) -> DataAwareRandomFeatures:
        """Redraw the base random projections in place."""
        if self.projection_is_fixed and not force:
            return self

        generation_device = self.projection_matrix.device
        if generator is not None:
            generation_device = generator.device
        projection = _gaussian_projection(
            self.num_features,
            self.rank,
            orthogonal=self.orthogonal,
            device=generation_device,
            generator=generator,
        )
        self.projection_matrix.copy_(
            projection.to(
                device=self.projection_matrix.device,
                dtype=self.projection_matrix.dtype,
            )
        )
        return self

    @torch.no_grad()
    def fix_projection_matrix_(self) -> DataAwareRandomFeatures:
        """Prevent ordinary projection redraws."""
        self._projection_fixed.fill_(True)
        return self

    @torch.no_grad()
    def unfix_projection_matrix_(self) -> DataAwareRandomFeatures:
        """Allow projection redraws."""
        self._projection_fixed.fill_(False)
        return self

    def covariance(self) -> torch.Tensor:
        """Return the positive-semidefinite covariance for every head."""
        covariance = self.geometry.transpose(-1, -2) @ self.geometry
        if not self.per_head:
            covariance = covariance.expand(self.num_heads, -1, -1)
        return covariance

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_mask: torch.Tensor | None,
    ) -> None:
        for name, tensor in (("query", query), ("key", key)):
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must have shape [batch, heads, length, head_dim]"
                )
            if tensor.shape[1] != self.num_heads:
                raise ValueError(
                    f"{name} has {tensor.shape[1]} heads, expected {self.num_heads}"
                )
            if tensor.shape[-1] != self.head_dim:
                raise ValueError(
                    f"{name} has head dimension {tensor.shape[-1]}, "
                    f"expected {self.head_dim}"
                )
        if query.shape[:2] != key.shape[:2]:
            raise ValueError("query and key batch and head dimensions must match")
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("query and key must share a device and dtype")
        if key_mask is not None:
            expected_shape = (key.shape[0], key.shape[2])
            if key_mask.dtype != torch.bool:
                raise TypeError("key_mask must have dtype torch.bool")
            if key_mask.device != key.device:
                raise ValueError("key_mask must be on the same device as key")
            if tuple(key_mask.shape) != expected_shape:
                raise ValueError(f"key_mask must have shape {expected_shape}")

    def _geometry_for_heads(self) -> torch.Tensor:
        if self.per_head:
            return self.geometry
        return self.geometry.expand(self.num_heads, -1, -1)

    def _validate_data(self, data: torch.Tensor) -> None:
        if data.ndim != 4:
            raise ValueError("data must have shape [batch, heads, length, head_dim]")
        if data.shape[1] != self.num_heads or data.shape[-1] != self.head_dim:
            raise ValueError(
                f"data must have {self.num_heads} heads and head_dim {self.head_dim}"
            )

    def feature_logits(self, data: torch.Tensor) -> torch.Tensor:
        """Return unnormalized log positive features."""
        self._validate_data(data)

        return self._feature_logits(data)

    def _feature_logits(self, data: torch.Tensor) -> torch.Tensor:
        scaled_data = data * (self.head_dim**-0.25)
        transformed = torch.matmul(
            scaled_data,
            self._geometry_for_heads().transpose(-1, -2),
        )
        projected = torch.matmul(
            transformed,
            self.projection_matrix.transpose(0, 1),
        )
        accumulation_dtype = (
            torch.float32
            if projected.dtype in (torch.float16, torch.bfloat16)
            else projected.dtype
        )
        projected = projected.to(accumulation_dtype)
        transformed = transformed.to(accumulation_dtype)
        return projected - 0.5 * transformed.square().sum(
            dim=-1,
            keepdim=True,
        )

    def _feature_map(
        self,
        data: torch.Tensor,
        *,
        is_query: bool,
        mask: torch.Tensor | None,
        stabilize: bool,
    ) -> torch.Tensor:
        logits = self._feature_logits(data)

        if stabilize:
            if is_query:
                maximum = logits.amax(dim=-1, keepdim=True).detach()
            elif mask is None:
                maximum = logits.amax(dim=(-2, -1), keepdim=True).detach()
            else:
                valid_logits = logits.masked_fill(
                    ~mask[:, None, :, None],
                    -torch.inf,
                )
                maximum = valid_logits.amax(dim=(-2, -1), keepdim=True).detach()
                maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
            maximum = maximum.clamp_min(0.0)
            features = torch.exp(logits - maximum)
            features = features + self.eps * torch.exp(-maximum)
        else:
            features = torch.exp(logits) + self.eps
        features = features * self.num_features**-0.5
        if mask is not None:
            features = features.masked_fill(~mask[:, None, :, None], 0.0)
        return features

    def query_feature_map(
        self,
        query: torch.Tensor,
        *,
        stabilize: bool = True,
    ) -> torch.Tensor:
        """Map queries into the learned random-feature space."""
        self._validate_data(query)
        return self._feature_map(
            query,
            is_query=True,
            mask=None,
            stabilize=stabilize,
        )

    def key_feature_map(
        self,
        key: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
        stabilize: bool = True,
    ) -> torch.Tensor:
        """Map keys into the learned random-feature space."""
        self._validate_data(key)
        if key_mask is not None:
            expected_shape = (key.shape[0], key.shape[2])
            if key_mask.dtype != torch.bool:
                raise TypeError("key_mask must have dtype torch.bool")
            if key_mask.device != key.device:
                raise ValueError("key_mask must be on the same device as key")
            if tuple(key_mask.shape) != expected_shape:
                raise ValueError(f"key_mask must have shape {expected_shape}")
        return self._feature_map(
            key,
            is_query=False,
            mask=key_mask,
            stabilize=stabilize,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        key_mask: torch.Tensor | None = None,
        stabilize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map queries and keys into the learned random-feature space."""
        self._validate_inputs(query, key, key_mask)
        query_features = self._feature_map(
            query,
            is_query=True,
            mask=None,
            stabilize=stabilize,
        )
        key_features = self._feature_map(
            key,
            is_query=False,
            mask=key_mask,
            stabilize=stabilize,
        )
        return query_features, key_features
