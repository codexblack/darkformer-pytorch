"""Data-aware positive random features."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

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
        orthogonal_block, upper = torch.linalg.qr(unstructured, mode="reduced")
        orthogonal_block = orthogonal_block * torch.sign(torch.diagonal(upper))
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
      eps: Nonnegative offset added to feature values.
      projection_seed: Optional seed for projection initialization.
      deterministic: Whether to start with a fixed projection matrix.
    """

    geometry: nn.Parameter
    projection_matrix: torch.Tensor
    _projection_fixed: torch.Tensor
    _projection_fixed_value: bool
    _proposal_active: torch.Tensor
    _proposal_active_value: bool
    _proposal_log_weights: torch.Tensor
    _proposal_projection: torch.Tensor
    _proposal_root: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_features: int,
        rank: int | None = None,
        per_head: bool = True,
        orthogonal: bool = True,
        eps: float = 0.0,
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
        if eps < 0.0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and nonnegative")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_features = num_features
        self.rank = rank
        self.per_head = per_head
        self.orthogonal = orthogonal
        self.eps = float(eps)
        self.projection_seed = projection_seed

        if rank < head_dim:
            warnings.warn(
                "rank-deficient geometry preserves the learned kernel estimator "
                "but not the full-density importance-sampling interpretation",
                UserWarning,
                stacklevel=2,
            )

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
        proposal_heads = geometry_heads
        proposal_root = torch.eye(rank).expand(proposal_heads, -1, -1).clone()
        self.register_buffer("_proposal_root", proposal_root, persistent=True)
        self._proposal_active_value = False
        self.register_buffer(
            "_proposal_active",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_proposal_projection",
            torch.empty(0, dtype=projection.dtype),
            persistent=False,
        )
        self.register_buffer(
            "_proposal_log_weights",
            torch.empty(0, dtype=projection.dtype),
            persistent=False,
        )
        self._projection_fixed_value = bool(deterministic)
        self.register_buffer(
            "_projection_fixed",
            torch.tensor(self._projection_fixed_value, dtype=torch.bool),
            persistent=True,
        )
        self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
            self._restore_missing_proposal_state
        )
        self.register_load_state_dict_post_hook(  # type: ignore[no-untyped-call]
            self._sync_runtime_state
        )

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> DataAwareRandomFeatures:
        super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]
        # Nonpersistent derived buffers are cast by ``Module._apply`` too.
        # Rebuilding restores float32 accumulation for low-precision modules
        # and keeps calibration/casting order from changing feature weights.
        self._refresh_proposal_cache()
        return self

    def _restore_missing_proposal_state(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: object,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_messages: list[str],
    ) -> None:
        del (
            module,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_messages,
        )
        root_key = f"{prefix}_proposal_root"
        active_key = f"{prefix}_proposal_active"
        if root_key not in state_dict:
            identity = torch.eye(
                self.rank,
                device=self._proposal_root.device,
                dtype=self._proposal_root.dtype,
            )
            state_dict[root_key] = identity.expand_as(self._proposal_root).clone()
        if active_key not in state_dict:
            state_dict[active_key] = torch.zeros_like(self._proposal_active)

    def _sync_runtime_state(
        self,
        module: nn.Module,
        incompatible_keys: object,
    ) -> None:
        del module, incompatible_keys
        self._projection_fixed_value = bool(self._projection_fixed.item())
        self._proposal_active_value = bool(self._proposal_active.item())
        self._refresh_proposal_cache()

    @property
    def projection_is_fixed(self) -> bool:
        """Whether ordinary redraw requests are disabled."""
        return self._projection_fixed_value

    @property
    def proposal_is_active(self) -> bool:
        """Whether projections use a separate importance-sampling proposal."""
        return self._proposal_active_value

    @torch.no_grad()
    def _refresh_proposal_cache(self) -> None:
        if not self.proposal_is_active:
            self._proposal_projection = self.projection_matrix.new_empty(0)
            self._proposal_log_weights = self.projection_matrix.new_empty(0)
            return

        proposal_projection = torch.matmul(
            self.projection_matrix.unsqueeze(0),
            self._proposal_root.transpose(-1, -2),
        )
        accumulation_dtype = (
            torch.float32
            if proposal_projection.dtype in (torch.float16, torch.bfloat16)
            else proposal_projection.dtype
        )
        base_projection = self.projection_matrix.to(accumulation_dtype)
        proposal_projection_accumulated = proposal_projection.to(accumulation_dtype)
        root = self._proposal_root.to(accumulation_dtype)
        sign, root_log_determinant = torch.linalg.slogdet(root)
        if bool((sign <= 0.0).any()):
            raise RuntimeError("proposal root must have positive determinant")
        log_weights = 0.5 * root_log_determinant[:, None]
        log_weights = log_weights + 0.25 * (
            base_projection.square().sum(dim=-1)[None, :]
            - proposal_projection_accumulated.square().sum(dim=-1)
        )
        self._proposal_projection = proposal_projection
        self._proposal_log_weights = log_weights

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
        self._refresh_proposal_cache()
        return self

    @torch.no_grad()
    def fix_projection_matrix_(self) -> DataAwareRandomFeatures:
        """Prevent ordinary projection redraws."""
        self._projection_fixed.fill_(True)
        self._projection_fixed_value = True
        return self

    @torch.no_grad()
    def unfix_projection_matrix_(self) -> DataAwareRandomFeatures:
        """Allow projection redraws."""
        self._projection_fixed.fill_(False)
        self._projection_fixed_value = False
        return self

    def covariance(self) -> torch.Tensor:
        """Return the positive-semidefinite covariance for every head."""
        covariance = self.geometry.transpose(-1, -2) @ self.geometry
        if not self.per_head:
            covariance = covariance.expand(self.num_heads, -1, -1)
        return covariance

    def proposal_covariance(self) -> torch.Tensor:
        """Return the projection proposal covariance for every head."""
        covariance = self._proposal_root @ self._proposal_root.transpose(-1, -2)
        if not self.per_head:
            covariance = covariance.expand(self.num_heads, -1, -1)
        return covariance

    @torch.no_grad()
    def reset_variance_optimal_proposal_(self) -> DataAwareRandomFeatures:
        """Reset projection sampling to the standard isotropic distribution."""
        identity = torch.eye(
            self.rank,
            device=self._proposal_root.device,
            dtype=self._proposal_root.dtype,
        )
        self._proposal_root.copy_(identity.expand_as(self._proposal_root))
        self._proposal_active.fill_(False)
        self._proposal_active_value = False
        self._refresh_proposal_cache()
        return self

    def _validate_calibration_mask(
        self,
        name: str,
        mask: torch.Tensor | None,
        data: torch.Tensor,
    ) -> None:
        if mask is None:
            return
        expected_shape = (data.shape[0], data.shape[2])
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")
        if mask.device != data.device:
            raise ValueError(f"{name} must be on the same device as its data")
        if tuple(mask.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    def _empirical_covariance(
        self,
        data: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        shared: bool,
        expected_dimension: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        expected_dimension = (
            self.head_dim if expected_dimension is None else expected_dimension
        )
        if data.ndim != 4:
            raise ValueError("data must have shape [batch, heads, length, dimension]")
        if data.shape[1] != self.num_heads or data.shape[-1] != expected_dimension:
            raise ValueError(
                f"data must have {self.num_heads} heads and dimension "
                f"{expected_dimension}"
            )
        if data.device != self.geometry.device:
            raise ValueError("calibration data must share the geometry device")
        accumulation_dtype = (
            torch.float64 if data.dtype == torch.float64 else torch.float32
        )
        samples = data.to(accumulation_dtype)
        if mask is None:
            if shared:
                count = data.shape[0] * data.shape[1] * data.shape[2]
                mean = samples.mean(dim=(0, 1, 2), keepdim=True)
            else:
                count = data.shape[0] * data.shape[2]
                mean = samples.mean(dim=(0, 2), keepdim=True)
            centered = samples - mean
        else:
            count = int(mask.sum().item()) * (data.shape[1] if shared else 1)
            weights = mask[:, None, :, None].to(accumulation_dtype)
            dimensions = (0, 1, 2) if shared else (0, 2)
            mean = (samples * weights).sum(
                dim=dimensions,
                keepdim=True,
            ) / max(count, 1)
            centered = (samples - mean) * weights
        if count < 2:
            raise ValueError("whitening requires at least two valid samples")
        if shared:
            covariance = torch.einsum(
                "bhld,bhle->de",
                centered,
                centered,
            ).unsqueeze(0)
        else:
            covariance = torch.einsum(
                "bhld,bhle->hde",
                centered,
                centered,
            )
        covariance = covariance / (count - 1)
        return covariance, count - 1

    @torch.no_grad()
    def initialize_variance_optimal_proposal_(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> DataAwareRandomFeatures:
        """Initialize the separate proposal from Theorem 3.2.

        The covariance is estimated after applying the feature map's
        ``head_dim**-0.25`` scaling and the current learned geometry. For a
        common centered Gaussian transformed input covariance ``Lambda``,
        projection vectors are then sampled from the variance-optimal proposal
        ``(I + 2 Lambda) @ inv(I - 2 Lambda)``. Each positive feature receives
        the square root of the exact ``p_I / p_proposal`` density ratio, so
        changing the proposal does not change the kernel being estimated.
        For unequal or non-Gaussian query/key distributions, the pooled
        moment-matched proposal remains unbiased but is not guaranteed to be
        variance-optimal. Orthogonal features likewise preserve the corrected
        marginal expectation, while the theorem's IID variance claim does not
        cover their inter-row dependence.

        Args:
          query: Projected queries with shape `[batch, heads, length, head_dim]`.
          key: Optional projected keys. When omitted, only queries are used.
          query_mask: Valid query positions with shape `[batch, length]`.
          key_mask: Valid key positions with shape `[batch, key_length]`.

        Returns:
          This module.

        Raises:
          ValueError: Inputs are invalid or the transformed covariance has an
            eigenvalue at least ``0.5``.
        """
        self._validate_data(query)
        if query.device != self.geometry.device:
            raise ValueError("calibration data must share the geometry device")
        self._validate_calibration_mask("query_mask", query_mask, query)
        accumulation_dtype = (
            torch.float64 if query.dtype == torch.float64 else torch.float32
        )
        geometry = self._geometry_for_heads().to(accumulation_dtype)
        normalizer = self.head_dim**-0.25
        transformed_query = torch.matmul(
            query.to(accumulation_dtype) * normalizer,
            geometry.transpose(-1, -2),
        )
        query_covariance, query_degrees = self._empirical_covariance(
            transformed_query,
            query_mask,
            shared=not self.per_head,
            expected_dimension=self.rank,
        )
        covariance = query_covariance
        degrees = query_degrees
        if key is not None:
            if key.device != query.device or key.dtype != query.dtype:
                raise ValueError("query and key must share a device and dtype")
            self._validate_data(key)
            self._validate_calibration_mask("key_mask", key_mask, key)
            transformed_key = torch.matmul(
                key.to(accumulation_dtype) * normalizer,
                geometry.transpose(-1, -2),
            )
            key_covariance, key_degrees = self._empirical_covariance(
                transformed_key,
                key_mask,
                shared=not self.per_head,
                expected_dimension=self.rank,
            )
            degrees += key_degrees
            covariance = (
                query_covariance * query_degrees + key_covariance * key_degrees
            ) / degrees
        elif key_mask is not None:
            raise ValueError("key_mask requires key calibration data")

        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        tolerance = (
            torch.finfo(eigenvalues.dtype).eps
            * self.rank
            * covariance.abs().amax().clamp_min(1.0)
        )
        if bool((eigenvalues < -tolerance).any()):
            raise ValueError(
                "transformed empirical covariance is not positive semidefinite"
            )
        eigenvalues = eigenvalues.clamp_min(0.0)
        if bool((eigenvalues >= 0.5).any()):
            raise ValueError(
                "variance-optimal proposal requires every transformed covariance "
                "eigenvalue to be below 0.5; reduce the geometry scale"
            )
        proposal_eigenvalues = (1.0 + 2.0 * eigenvalues) / (1.0 - 2.0 * eigenvalues)
        proposal_root = (
            eigenvectors
            @ torch.diag_embed(proposal_eigenvalues.sqrt())
            @ eigenvectors.transpose(-1, -2)
        )
        self._proposal_root.copy_(proposal_root.to(self._proposal_root))
        self._proposal_active.fill_(True)
        self._proposal_active_value = True
        self._refresh_proposal_cache()
        return self

    @torch.no_grad()
    def initialize_whitening_(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        regularization: float = 1e-4,
        shrinkage: float = 0.0,
        geometry_scale: float | None = None,
    ) -> DataAwareRandomFeatures:
        """Initialize geometry from empirical query and key covariance.

        The full-rank geometry is set to the symmetric inverse square root of
        the pooled within-query and within-key covariance. Regularization is
        relative to the mean marginal variance.

        Args:
          query: Projected queries with shape `[batch, heads, length, head_dim]`.
          key: Optional projected keys. When omitted, only queries are used.
          query_mask: Valid query positions with shape `[batch, length]`.
          key_mask: Valid key positions with shape `[batch, key_length]`.
          regularization: Nonnegative diagonal loading relative to mean variance.
          shrinkage: Weight assigned to an isotropic covariance target.
          geometry_scale: Positive scale applied after inverse-covariance
            whitening. Defaults to ``head_dim**-0.25``. At this low-level API,
            inputs are whitened exactly as passed; use kernel-scaled calibration
            tensors to reproduce the high-level temperature-preserving policy.
            Pass ``1.0`` for literal whitening.

        Returns:
          This module.

        Warning:
          Literal whitening targets unit covariance and can make exponential
          random-feature variance impractical at ordinary head dimensions.

        Raises:
          ValueError: Inputs are invalid or the geometry is rank deficient.
        """
        if self.rank != self.head_dim:
            raise ValueError("whitening initialization requires full-rank geometry")
        default_geometry_scale = self.head_dim**-0.25
        using_default_scale = geometry_scale is None
        if geometry_scale is None:
            geometry_scale = default_geometry_scale
        if regularization < 0.0 or not math.isfinite(regularization):
            raise ValueError("regularization must be finite and nonnegative")
        if not 0.0 <= shrinkage <= 1.0 or not math.isfinite(shrinkage):
            raise ValueError("shrinkage must be finite and in [0, 1]")
        if geometry_scale <= 0.0 or not math.isfinite(geometry_scale):
            raise ValueError("geometry_scale must be finite and positive")
        using_default_scale = (
            using_default_scale or geometry_scale == default_geometry_scale
        )
        self._validate_calibration_mask("query_mask", query_mask, query)
        query_covariance, query_degrees = self._empirical_covariance(
            query,
            query_mask,
            shared=not self.per_head,
        )
        covariance = query_covariance
        degrees = query_degrees
        if key is not None:
            if key.device != query.device or key.dtype != query.dtype:
                raise ValueError("query and key must share a device and dtype")
            self._validate_calibration_mask("key_mask", key_mask, key)
            key_covariance, key_degrees = self._empirical_covariance(
                key,
                key_mask,
                shared=not self.per_head,
            )
            degrees += key_degrees
            covariance = (
                query_covariance * query_degrees + key_covariance * key_degrees
            ) / degrees
        elif key_mask is not None:
            raise ValueError("key_mask requires key calibration data")

        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        scale = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
        scale = scale.clamp_min(torch.finfo(covariance.dtype).eps)
        identity = torch.eye(
            self.head_dim,
            device=covariance.device,
            dtype=covariance.dtype,
        ).expand(covariance.shape[0], -1, -1)
        covariance = (
            (1.0 - shrinkage) * covariance
            + shrinkage * scale[:, None, None] * identity
            + regularization * scale[:, None, None] * identity
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        if bool((eigenvalues <= 0.0).any()):
            raise ValueError(
                "empirical covariance is not positive definite; increase "
                "regularization or shrinkage"
            )
        inverse_root = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.rsqrt())
            @ eigenvectors.transpose(-1, -2)
        )
        expected_squared_norm = self.head_dim * geometry_scale**2
        if using_default_scale:
            warning = None
        elif geometry_scale == 1.0:
            warning = (
                "literal whitening targets unit transformed covariance, so the "
                f"expected squared feature input norm is approximately head_dim="
                f"{self.head_dim}; positive random-feature variance can grow "
                "exponentially with this dimension; consider geometry_scale="
                f"{self.head_dim**-0.25:g} to preserve the usual score temperature "
                f"or geometry_scale={self.head_dim**-0.5:g} to target unit expected "
                "squared norm"
            )
        else:
            warning = (
                f"scaled whitening with geometry_scale={geometry_scale:g} targets "
                f"transformed covariance {geometry_scale**2:g} * I, so the expected "
                "squared feature input norm is approximately "
                f"{expected_squared_norm:g}; this scale changes both attention "
                "temperature and positive random-feature variance"
            )
        if warning is not None:
            warnings.warn(warning, UserWarning, stacklevel=3)
        self.geometry.copy_((geometry_scale * inverse_root).to(self.geometry))
        return self

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
        proposal_log_weights = None
        if self.proposal_is_active:
            proposal_projection = self._proposal_projection
            proposal_log_weights = self._proposal_log_weights
            if not self.per_head:
                proposal_projection = proposal_projection.expand(
                    self.num_heads,
                    -1,
                    -1,
                )
                proposal_log_weights = proposal_log_weights.expand(
                    self.num_heads,
                    -1,
                )
            projected = torch.einsum(
                "bhld,hmd->bhlm",
                transformed,
                proposal_projection,
            )
        else:
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
        logits = projected - 0.5 * transformed.square().sum(
            dim=-1,
            keepdim=True,
        )
        if proposal_log_weights is not None:
            return logits + proposal_log_weights[None, :, None, :].to(
                accumulation_dtype
            )
        return logits

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
            # A negative shift is safe when there is no additive feature floor
            # and is required to keep all-negative logits representable. With a
            # positive floor it could make eps * exp(-maximum) overflow.
            if self.eps > 0.0:
                maximum = maximum.clamp_min(0.0)
            stabilized_logits = logits - maximum
            if data.dtype in (torch.float16, torch.bfloat16):
                # Compute the norm correction and stabilizing maximum in fp32,
                # but keep the feature-sized exponential in the model dtype.
                # Its argument is nonpositive, so this cannot introduce the
                # low-precision overflow that the fp32 logit path prevents.
                stabilized_logits = stabilized_logits.to(data.dtype)
            features = torch.exp(stabilized_logits)
            if self.eps > 0.0:
                features = features + self.eps * torch.exp(-maximum.to(features.dtype))
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
        """Map queries into the learned random-feature space.

        Stabilization rescales each query feature vector by a positive scalar.
        This leaves normalized attention unchanged. Set ``stabilize=False``
        when using feature inner products as an unbiased kernel estimator.
        """
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
        """Map keys into the learned random-feature space.

        Stabilization applies one positive scale per batch and head. This leaves
        normalized attention unchanged. Set ``stabilize=False`` when using
        feature inner products as an unbiased kernel estimator.
        """
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
