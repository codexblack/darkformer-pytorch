"""Benchmark exact and random-feature attention implementations."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils import benchmark

from darkformer_pytorch import DarkformerKernelAttention, __version__

MethodName = Literal["sdpa-math", "sdpa-flash", "performer", "darkformer"]
DtypeName = Literal["auto", "float32", "float16", "bfloat16"]
FeatureStructure = Literal["iid", "orthogonal"]
PerformerAccumulation = Literal["input", "float32"]
ContextFactory = Callable[[], AbstractContextManager[None]]
Operation = Callable[[], torch.Tensor]
TensorFunction = Callable[..., torch.Tensor]


@dataclass(slots=True)
class Method:
    """An attention operation and its dispatch context."""

    name: MethodName
    operation: Operation
    context: ContextFactory
    accumulation_precision: str


@dataclass(slots=True)
class PerformanceResult:
    """One latency and memory measurement."""

    sequence_length: int
    method: str
    accumulation_precision: str
    latency_median_ms: float | None
    latency_iqr_ms: float | None
    latency_repeat_medians_ms: list[float]
    tokens_per_second: float | None
    incremental_peak_memory_mib: float | None
    timing_blocks: int
    status: str


@dataclass(slots=True)
class ErrorResult:
    """Approximation error across projection seeds."""

    sequence_length: int
    method: str
    reference: str
    relative_l2_median: float | None
    relative_l2_iqr: float | None
    relative_l2_samples: list[float]
    projection_seeds: int
    status: str


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0.0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return number


def _condition_number(value: str) -> float:
    number = float(value)
    if number < 1.0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("value must be finite and at least 1")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def _median_iqr(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    median = statistics.median(values)
    if len(values) == 1:
        return median, 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return median, quartiles[2] - quartiles[0]


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA is not available")
    return device


def _dtype(name: DtypeName, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _null_context() -> AbstractContextManager[None]:
    return nullcontext()


def _sdpa_context(backend: SDPBackend) -> ContextFactory:
    def context() -> AbstractContextManager[None]:
        return cast(AbstractContextManager[None], sdpa_kernel(backend))

    return context


def _flash_attention_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    availability_check = getattr(
        torch.backends.cuda,
        "is_flash_attention_available",
        None,
    )
    return availability_check is None or bool(availability_check())


def _projection_matrix(
    rows: int,
    columns: int,
    structure: FeatureStructure,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    if structure == "iid":
        return torch.randn(rows, columns, generator=generator)

    blocks = []
    remaining = rows
    while remaining > 0:
        unstructured = torch.randn(
            columns,
            columns,
            generator=generator,
        )
        orthogonal, upper = torch.linalg.qr(unstructured, mode="reduced")
        orthogonal = orthogonal * torch.sign(torch.diagonal(upper))
        block_rows = min(remaining, columns)
        blocks.append(orthogonal.transpose(0, 1)[:block_rows])
        remaining -= block_rows
    directions = torch.cat(blocks, dim=0)
    row_norms = torch.randn(
        rows,
        columns,
        generator=generator,
    ).norm(dim=1)
    return directions * row_norms[:, None]


class _ControlledPerformerAttention(nn.Module):
    """Performer attention with explicit projection and epsilon controls."""

    projection_matrix: torch.Tensor

    def __init__(
        self,
        projection_matrix: torch.Tensor,
        *,
        head_dim: int,
        causal: bool,
        eps: float,
        accumulation: PerformerAccumulation,
    ) -> None:
        super().__init__()
        try:
            module = importlib.import_module("performer_pytorch.performer_pytorch")
        except ImportError as error:
            raise RuntimeError(
                "performer-pytorch is required for the Performer benchmark"
            ) from error
        self._kernel = cast(TensorFunction, module.softmax_kernel)
        if causal:
            attention_type = getattr(module, "FastAttention", None)
            if attention_type is None:
                raise RuntimeError("performer_pytorch.FastAttention is unavailable")
            attention = attention_type(
                dim_heads=head_dim,
                nb_features=projection_matrix.shape[0],
                causal=True,
            )
            self._attention = cast(
                TensorFunction,
                attention.causal_linear_fn,
            )
        else:
            self._attention = cast(
                TensorFunction,
                module.linear_attention,
            )
        self.eps = eps
        self.accumulation = accumulation
        self.register_buffer(
            "projection_matrix",
            projection_matrix,
            persistent=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply controlled positive random-feature attention."""
        output_dtype = value.dtype
        if self.accumulation == "float32":
            query = query.float()
            key = key.float()
            value = value.float()
        query_features = self._kernel(
            query,
            projection_matrix=self.projection_matrix,
            is_query=True,
            eps=self.eps,
        )
        key_features = self._kernel(
            key,
            projection_matrix=self.projection_matrix,
            is_query=False,
            eps=self.eps,
        )
        return self._attention(query_features, key_features, value).to(output_dtype)


def _performer_attention(
    head_dim: int,
    causal: bool,
    device: torch.device,
    dtype: torch.dtype,
    projection_matrix: torch.Tensor,
    eps: float,
    accumulation: PerformerAccumulation,
) -> nn.Module:
    projection_dtype = torch.float32 if accumulation == "float32" else dtype
    projection = projection_matrix.to(
        device=device,
        dtype=projection_dtype,
    )
    return _ControlledPerformerAttention(
        projection,
        head_dim=head_dim,
        causal=causal,
        eps=eps,
        accumulation=accumulation,
    ).eval()


def _performer_accumulation_label(
    accumulation: PerformerAccumulation,
    dtype: torch.dtype,
) -> str:
    if accumulation == "float32":
        return "float32 features and reductions"
    name = str(dtype).removeprefix("torch.")
    return f"{name} features and reductions"


def _darkformer_accumulation_label(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    name = str(dtype).removeprefix("torch.")
    return f"{name} stabilized features and reductions; float32 logits and scales"


def _darkformer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    calibration_query: torch.Tensor,
    calibration_key: torch.Tensor,
    *,
    num_features: int,
    causal: bool,
    seed: int,
    mode: Literal["linear", "exact"],
    projection_matrix: torch.Tensor,
    feature_structure: FeatureStructure,
    eps: float,
    regularization: float,
    shrinkage: float,
    geometry_scale: float,
    causal_chunk_size: int,
) -> DarkformerKernelAttention:
    attention = DarkformerKernelAttention(
        query.shape[-1],
        query.shape[1],
        num_features=num_features,
        orthogonal_features=feature_structure == "orthogonal",
        causal=causal,
        attention_mode=mode,
        exact_backend="sdpa",
        causal_chunk_size=causal_chunk_size,
        eps=eps,
        projection_seed=seed,
        deterministic=True,
    ).to(device=query.device, dtype=query.dtype)
    attention.eval().fix_projection_matrices_()
    with torch.no_grad():
        attention.random_features.projection_matrix.copy_(
            projection_matrix.to(
                device=query.device,
                dtype=query.dtype,
            )
        )
    attention.initialize_whitening_(
        calibration_query,
        calibration_key,
        regularization=regularization,
        shrinkage=shrinkage,
        geometry_scale=geometry_scale,
    )
    return attention


def _method(
    name: MethodName,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    calibration_query: torch.Tensor,
    calibration_key: torch.Tensor,
    *,
    num_features: int,
    causal: bool,
    seed: int,
    projection_matrix: torch.Tensor,
    feature_structure: FeatureStructure,
    estimator_eps: float,
    performer_accumulation: PerformerAccumulation,
    regularization: float,
    shrinkage: float,
    geometry_scale: float,
    causal_chunk_size: int,
) -> Method:
    if name in ("sdpa-math", "sdpa-flash"):
        if name == "sdpa-flash" and not _flash_attention_available(query.device):
            raise RuntimeError("PyTorch was not compiled with FlashAttention")
        backend = SDPBackend.MATH if name == "sdpa-math" else SDPBackend.FLASH_ATTENTION
        return Method(
            name,
            lambda: functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=causal,
            ),
            _sdpa_context(backend),
            "backend managed",
        )
    if name == "performer":
        attention = _performer_attention(
            query.shape[-1],
            causal,
            query.device,
            query.dtype,
            projection_matrix,
            estimator_eps,
            performer_accumulation,
        )
        return Method(
            name,
            lambda: attention(query, key, value),
            _null_context,
            _performer_accumulation_label(
                performer_accumulation,
                query.dtype,
            ),
        )
    attention = _darkformer_attention(
        query,
        key,
        calibration_query,
        calibration_key,
        num_features=num_features,
        causal=causal,
        seed=seed,
        mode="linear",
        projection_matrix=projection_matrix,
        feature_structure=feature_structure,
        eps=estimator_eps,
        regularization=regularization,
        shrinkage=shrinkage,
        geometry_scale=geometry_scale,
        causal_chunk_size=causal_chunk_size,
    )
    return Method(
        name,
        lambda: attention(query, key, value),
        _null_context,
        _darkformer_accumulation_label(query.dtype),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _failure_status(error: RuntimeError) -> str:
    message = str(error).strip().splitlines()[0]
    if "out of memory" in message.lower():
        return "out of memory"
    return f"unavailable: {message}"


def _measure(
    method: Method,
    *,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    warmup: int,
    min_run_time: float,
    timing_repeats: int,
) -> PerformanceResult:
    try:
        with torch.inference_mode(), method.context():
            for _ in range(warmup):
                method.operation()
            _synchronize(device)

            incremental_peak_memory_mib = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                output = method.operation()
                _synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                incremental_peak_memory_mib = (peak - baseline) / (1024.0**2)
                del output

            timer = benchmark.Timer(
                stmt="operation()",
                globals={"operation": method.operation},
            )
            repeat_medians = []
            timing_blocks = 0
            for _ in range(timing_repeats):
                measurement = timer.blocked_autorange(min_run_time=min_run_time)
                repeat_medians.append(measurement.median)
                timing_blocks += len(measurement.raw_times)
            seconds, seconds_iqr = _median_iqr(repeat_medians)
        return PerformanceResult(
            sequence_length=sequence_length,
            method=method.name,
            accumulation_precision=method.accumulation_precision,
            latency_median_ms=seconds * 1_000.0,
            latency_iqr_ms=seconds_iqr * 1_000.0,
            latency_repeat_medians_ms=[repeat * 1_000.0 for repeat in repeat_medians],
            tokens_per_second=batch_size * sequence_length / seconds,
            incremental_peak_memory_mib=incremental_peak_memory_mib,
            timing_blocks=timing_blocks,
            status="ok",
        )
    except RuntimeError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return PerformanceResult(
            sequence_length=sequence_length,
            method=method.name,
            accumulation_precision=method.accumulation_precision,
            latency_median_ms=None,
            latency_iqr_ms=None,
            latency_repeat_medians_ms=[],
            tokens_per_second=None,
            incremental_peak_memory_mib=None,
            timing_blocks=0,
            status=_failure_status(error),
        )


def _relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(output.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return float((difference / denominator).item())


def _error_reference(method_name: MethodName) -> str:
    if method_name == "darkformer":
        return "exact held-out-calibrated Mahalanobis attention"
    return "isotropic softmax (PyTorch SDPA math)"


def _error_result(
    method_name: MethodName,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    calibration_query: torch.Tensor,
    calibration_key: torch.Tensor,
    *,
    num_features: int,
    causal: bool,
    seeds: int,
    projection_seed: int,
    feature_structure: FeatureStructure,
    estimator_eps: float,
    performer_accumulation: PerformerAccumulation,
    regularization: float,
    shrinkage: float,
    geometry_scale: float,
    causal_chunk_size: int,
) -> ErrorResult:
    errors: list[float] = []
    reference_name = _error_reference(method_name)
    try:
        if method_name == "sdpa-flash" and not _flash_attention_available(query.device):
            raise RuntimeError("PyTorch was not compiled with FlashAttention")
        if method_name in ("sdpa-flash", "performer"):
            with torch.inference_mode(), _sdpa_context(SDPBackend.MATH)():
                reference = functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=causal,
                )
        if method_name == "sdpa-flash":
            with (
                torch.inference_mode(),
                _sdpa_context(SDPBackend.FLASH_ATTENTION)(),
            ):
                output = functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=causal,
                )
            errors.append(_relative_l2(output, reference))
        elif method_name == "performer":
            for seed_offset in range(seeds):
                seed = projection_seed + seed_offset
                projection = _projection_matrix(
                    num_features,
                    query.shape[-1],
                    feature_structure,
                    seed,
                )
                performer = _performer_attention(
                    query.shape[-1],
                    causal,
                    query.device,
                    query.dtype,
                    projection,
                    estimator_eps,
                    performer_accumulation,
                )
                with torch.inference_mode():
                    output = performer(query, key, value)
                errors.append(_relative_l2(output, reference))
        else:
            reference_projection = _projection_matrix(
                num_features,
                query.shape[-1],
                feature_structure,
                projection_seed,
            )
            exact = _darkformer_attention(
                query,
                key,
                calibration_query,
                calibration_key,
                num_features=num_features,
                causal=causal,
                seed=projection_seed,
                mode="exact",
                projection_matrix=reference_projection,
                feature_structure=feature_structure,
                eps=estimator_eps,
                regularization=regularization,
                shrinkage=shrinkage,
                geometry_scale=geometry_scale,
                causal_chunk_size=causal_chunk_size,
            )
            with torch.inference_mode():
                reference = exact(query, key, value)
            for seed_offset in range(seeds):
                seed = projection_seed + seed_offset
                projection = _projection_matrix(
                    num_features,
                    query.shape[-1],
                    feature_structure,
                    seed,
                )
                linear = _darkformer_attention(
                    query,
                    key,
                    calibration_query,
                    calibration_key,
                    num_features=num_features,
                    causal=causal,
                    seed=seed,
                    mode="linear",
                    projection_matrix=projection,
                    feature_structure=feature_structure,
                    eps=estimator_eps,
                    regularization=regularization,
                    shrinkage=shrinkage,
                    geometry_scale=geometry_scale,
                    causal_chunk_size=causal_chunk_size,
                )
                with torch.inference_mode():
                    output = linear(query, key, value)
                errors.append(_relative_l2(output, reference))
        median, iqr = _median_iqr(errors)
        return ErrorResult(
            sequence_length=query.shape[2],
            method=method_name,
            reference=reference_name,
            relative_l2_median=median,
            relative_l2_iqr=iqr,
            relative_l2_samples=errors,
            projection_seeds=1 if method_name == "sdpa-flash" else seeds,
            status="ok",
        )
    except RuntimeError as error:
        return ErrorResult(
            sequence_length=query.shape[2],
            method=method_name,
            reference=reference_name,
            relative_l2_median=None,
            relative_l2_iqr=None,
            relative_l2_samples=[],
            projection_seeds=0,
            status=_failure_status(error),
        )


def _stream_seed(base: int, stream: int, value: int = 0) -> int:
    return (base + 1_000_003 * stream + 9_176 * value) % (2**63 - 1)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _anisotropic_transform(
    head_dim: int,
    condition_number: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    unstructured = torch.randn(
        head_dim,
        head_dim,
        device=device,
        generator=generator,
    )
    basis, _ = torch.linalg.qr(unstructured)
    scales = torch.logspace(
        0.0,
        0.5 * math.log10(condition_number),
        head_dim,
        device=device,
    )
    scales = scales * scales.square().mean().rsqrt()
    return basis @ torch.diag(scales) @ basis.transpose(0, 1)


def _anisotropic_query_key(
    batch_size: int,
    heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    transform: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch_size, heads, sequence_length, head_dim)
    query = (
        torch.randn(
            shape,
            device=device,
            generator=generator,
        )
        @ transform
    )
    key = (
        torch.randn(
            shape,
            device=device,
            generator=generator,
        )
        @ transform
    )
    return query.to(dtype), key.to(dtype)


def _anisotropic_inputs(
    batch_size: int,
    heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    transform: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key = _anisotropic_query_key(
        batch_size,
        heads,
        sequence_length,
        head_dim,
        device,
        dtype,
        transform,
        generator,
    )
    shape = (batch_size, heads, sequence_length, head_dim)
    value = torch.randn(
        shape,
        device=device,
        generator=generator,
    ).to(dtype)
    return query, key, value


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _nvidia_driver(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    drivers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return drivers[0] if drivers else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "darkformer_pytorch": __version__,
        "darkformer_distribution": _package_version("darkformer-pytorch"),
        "performer_pytorch": _package_version("performer-pytorch"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "cuda": torch.version.cuda,
        "nvidia_driver": _nvidia_driver(device),
        "dtype": str(dtype).removeprefix("torch."),
        "memory_metric": "incremental peak allocation during one warmed forward",
    }


def _cell(value: float | None, digits: int) -> str:
    return "N/A" if value is None else f"{value:,.{digits}f}"


def _performance_table(results: Sequence[PerformanceResult]) -> str:
    lines = [
        "| Sequence | Method | Accumulation | Median latency (ms) | "
        "IQR (ms) | Tokens/s | Incremental peak allocation (MiB) | Status |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for result in results:
        if result.status == "ok":
            latency = _cell(result.latency_median_ms, 3)
            latency_iqr = _cell(result.latency_iqr_ms, 3)
            throughput = _cell(result.tokens_per_second, 0)
            memory = _cell(result.incremental_peak_memory_mib, 1)
        else:
            latency = latency_iqr = throughput = memory = "N/A"
        lines.append(
            f"| {result.sequence_length:,} | {result.method} | "
            f"{result.accumulation_precision} | {latency} | {latency_iqr} | "
            f"{throughput} | {memory} | {result.status} |"
        )
    return "\n".join(lines)


def _error_table(results: Sequence[ErrorResult]) -> str:
    lines = [
        "| Sequence | Method | Reference | Projection seeds | "
        "Median relative L2 | IQR | Status |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for result in results:
        median = _cell(result.relative_l2_median, 6)
        iqr = _cell(result.relative_l2_iqr, 6)
        lines.append(
            f"| {result.sequence_length:,} | {result.method} | "
            f"{result.reference} | {result.projection_seeds} | {median} | {iqr} | "
            f"{result.status} |"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SDPA, fused FlashAttention, controlled Performer, and "
            "calibrated DARKformer attention."
        )
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("sdpa-math", "sdpa-flash", "performer", "darkformer"),
        default=("sdpa-math", "sdpa-flash", "performer", "darkformer"),
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=_positive_int,
        default=(512, 1024, 2048, 4096),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--heads", type=_positive_int, default=8)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument("--num-features", type=_positive_int, default=256)
    parser.add_argument("--calibration-length", type=_positive_int, default=512)
    parser.add_argument("--warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--min-run-time", type=_positive_float, default=0.25)
    parser.add_argument("--timing-repeats", type=_positive_int, default=5)
    parser.add_argument("--error-length", type=_positive_int, default=512)
    parser.add_argument("--error-seeds", type=_positive_int, default=30)
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--projection-seed", type=_nonnegative_int, default=1_000)
    parser.add_argument(
        "--feature-structure",
        choices=("iid", "orthogonal"),
        default="orthogonal",
    )
    parser.add_argument(
        "--estimator-eps",
        type=_nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--performer-accumulation",
        choices=("input", "float32"),
        default="input",
    )
    parser.add_argument(
        "--condition-number",
        type=_condition_number,
        default=16.0,
    )
    parser.add_argument(
        "--regularization",
        type=_nonnegative_float,
        default=1e-4,
    )
    parser.add_argument("--shrinkage", type=_nonnegative_float, default=0.01)
    parser.add_argument("--geometry-scale", type=_positive_float, default=None)
    parser.add_argument("--causal-chunk-size", type=_positive_int, default=256)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/attention.json"),
    )
    return parser


def main() -> None:
    """Run the attention benchmarks and write ignored raw results."""
    args = _parser().parse_args()
    device = _device(cast(str, args.device))
    dtype = _dtype(cast(DtypeName, args.dtype), device)
    methods = cast(tuple[MethodName, ...], tuple(args.methods))
    lengths = cast(tuple[int, ...], tuple(args.sequence_lengths))
    feature_structure = cast(FeatureStructure, args.feature_structure)
    performer_accumulation = cast(
        PerformerAccumulation,
        args.performer_accumulation,
    )
    geometry_scale = (
        args.head_dim**-0.25
        if args.geometry_scale is None
        else cast(float, args.geometry_scale)
    )
    if args.shrinkage > 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if args.estimator_eps > 0.0 and "darkformer" in methods:
        raise ValueError("DARKformer approximation error requires --estimator-eps=0")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    metadata = _metadata(device, dtype)
    print(json.dumps(metadata, indent=2), flush=True)

    transform = _anisotropic_transform(
        args.head_dim,
        args.condition_number,
        device,
        _generator(device, _stream_seed(args.seed, 0)),
    )
    calibration_query, calibration_key = _anisotropic_query_key(
        args.batch_size,
        args.heads,
        args.calibration_length,
        args.head_dim,
        device,
        dtype,
        transform,
        _generator(device, _stream_seed(args.seed, 1)),
    )
    performance_projection = _projection_matrix(
        args.num_features,
        args.head_dim,
        feature_structure,
        args.projection_seed,
    )

    performance: list[PerformanceResult] = []
    for length in lengths:
        query, key, value = _anisotropic_inputs(
            args.batch_size,
            args.heads,
            length,
            args.head_dim,
            device,
            dtype,
            transform,
            _generator(device, _stream_seed(args.seed, 2, length)),
        )
        for method_name in methods:
            try:
                selected = _method(
                    method_name,
                    query,
                    key,
                    value,
                    calibration_query,
                    calibration_key,
                    num_features=args.num_features,
                    causal=args.causal,
                    seed=args.projection_seed,
                    projection_matrix=performance_projection,
                    feature_structure=feature_structure,
                    estimator_eps=args.estimator_eps,
                    performer_accumulation=performer_accumulation,
                    regularization=args.regularization,
                    shrinkage=args.shrinkage,
                    geometry_scale=geometry_scale,
                    causal_chunk_size=args.causal_chunk_size,
                )
                result = _measure(
                    selected,
                    sequence_length=length,
                    batch_size=args.batch_size,
                    device=device,
                    warmup=args.warmup,
                    min_run_time=args.min_run_time,
                    timing_repeats=args.timing_repeats,
                )
            except RuntimeError as error:
                result = PerformanceResult(
                    sequence_length=length,
                    method=method_name,
                    accumulation_precision="unavailable",
                    latency_median_ms=None,
                    latency_iqr_ms=None,
                    latency_repeat_medians_ms=[],
                    tokens_per_second=None,
                    incremental_peak_memory_mib=None,
                    timing_blocks=0,
                    status=_failure_status(error),
                )
            performance.append(result)
            print(
                f"length={length} method={method_name} status={result.status}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    error_dtype = torch.float32
    error_transform = transform.to(error_dtype)
    error_query, error_key, error_value = _anisotropic_inputs(
        args.batch_size,
        args.heads,
        args.error_length,
        args.head_dim,
        device,
        error_dtype,
        error_transform,
        _generator(device, _stream_seed(args.seed, 3)),
    )
    error_calibration_query, error_calibration_key = _anisotropic_query_key(
        args.batch_size,
        args.heads,
        args.calibration_length,
        args.head_dim,
        device,
        error_dtype,
        error_transform,
        _generator(device, _stream_seed(args.seed, 4)),
    )
    errors = [
        _error_result(
            method,
            error_query,
            error_key,
            error_value,
            error_calibration_query,
            error_calibration_key,
            num_features=args.num_features,
            causal=args.causal,
            seeds=args.error_seeds,
            projection_seed=args.projection_seed,
            feature_structure=feature_structure,
            estimator_eps=args.estimator_eps,
            performer_accumulation=performer_accumulation,
            regularization=args.regularization,
            shrinkage=args.shrinkage,
            geometry_scale=geometry_scale,
            causal_chunk_size=args.causal_chunk_size,
        )
        for method in methods
        if method != "sdpa-math"
    ]

    output = cast(Path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "config": {
            "protocol_version": 5,
            "methods": list(methods),
            "sequence_lengths": list(lengths),
            "batch_size": args.batch_size,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "num_features": args.num_features,
            "feature_structure": feature_structure,
            "estimator_eps": args.estimator_eps,
            "performer_accumulation": performer_accumulation,
            "causal": args.causal,
            "causal_chunk_size": args.causal_chunk_size,
            "condition_number": args.condition_number,
            "calibration_length": args.calibration_length,
            "regularization": args.regularization,
            "shrinkage": args.shrinkage,
            "geometry_scale": geometry_scale,
            "warmup": args.warmup,
            "min_run_time": args.min_run_time,
            "timing_repeats": args.timing_repeats,
            "error_length": args.error_length,
            "error_dtype": "float32",
            "error_seeds": args.error_seeds,
            "seed": args.seed,
            "projection_seed": args.projection_seed,
            "calibration_and_evaluation_inputs_are_disjoint": True,
        },
        "performance": [asdict(result) for result in performance],
        "approximation_error": [asdict(result) for result in errors],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\nPerformance\n", flush=True)
    print(_performance_table(performance), flush=True)
    print("\nApproximation error\n", flush=True)
    print(_error_table(errors), flush=True)
    print(f"\nRaw results: {output}", flush=True)


if __name__ == "__main__":
    main()
