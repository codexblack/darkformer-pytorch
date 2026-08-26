"""Benchmark DARKformer self-attention implementations."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import Literal, cast

import torch

from darkformer_pytorch import DarkformerAttention

AttentionMode = Literal["linear", "exact", "auto"]
DtypeName = Literal["auto", "float32", "float16", "bfloat16"]
ExactBackend = Literal["auto", "flash3", "flash2", "sdpa"]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def _sequence_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "sequence lengths must be comma-separated integers"
        ) from error
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return lengths


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
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    attention: DarkformerAttention,
    inputs: torch.Tensor,
    mask: torch.Tensor | None,
    warmup: int,
    iterations: int,
) -> tuple[float, float | None]:
    device = inputs.device
    with torch.inference_mode():
        for _ in range(warmup):
            attention(inputs, mask=mask)
        _synchronize(device)

        baseline_memory = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)

        start = time.perf_counter()
        for _ in range(iterations):
            attention(inputs, mask=mask)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1_000.0 / iterations
    if device.type != "cuda":
        return latency_ms, None
    peak_bytes = torch.cuda.max_memory_allocated(device) - baseline_memory
    return latency_ms, peak_bytes / (1024.0 * 1024.0)


def _format_table(rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.rjust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare DARKformer linear, exact, and automatic attention."
    )
    parser.add_argument("--device", default="auto", help="Device name or 'auto'.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("linear", "exact", "auto"),
        default=("linear", "exact", "auto"),
    )
    parser.add_argument(
        "--sequence-lengths",
        type=_sequence_lengths,
        default=(512, 2048, 4096),
        metavar="N[,N...]",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--dim", type=_positive_int, default=512)
    parser.add_argument("--heads", type=_positive_int, default=8)
    parser.add_argument("--head-dim", type=_positive_int, default=64)
    parser.add_argument("--num-features", type=_positive_int, default=256)
    parser.add_argument("--exact-threshold", type=_positive_int, default=1024)
    parser.add_argument(
        "--exact-backend",
        choices=("auto", "flash3", "flash2", "sdpa"),
        default="auto",
    )
    parser.add_argument("--causal-chunk-size", type=_positive_int, default=64)
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=50)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Benchmark a padding mask. External FlashAttention then falls back to SDPA."
        ),
    )
    return parser


def main() -> None:
    """Run the attention benchmark."""
    args = _parser().parse_args()
    device = _device(cast(str, args.device))
    dtype_name = cast(DtypeName, args.dtype)
    dtype = _dtype(dtype_name, device)
    modes = cast(tuple[AttentionMode, ...], tuple(args.modes))
    exact_backend = cast(ExactBackend, args.exact_backend)
    sequence_lengths = cast(tuple[int, ...], args.sequence_lengths)

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    rows = [["mode", "length", "latency (ms)", "tokens/s", "peak MiB"]]
    for length in sequence_lengths:
        inputs = torch.randn(
            args.batch_size,
            length,
            args.dim,
            device=device,
            dtype=dtype,
        )
        mask = None
        if args.mask:
            mask = torch.ones(
                args.batch_size,
                length,
                device=device,
                dtype=torch.bool,
            )
            mask[:, -(max(1, length // 8)) :] = False

        for mode in modes:
            torch.manual_seed(1)
            attention = DarkformerAttention(
                args.dim,
                heads=args.heads,
                head_dim=args.head_dim,
                num_features=args.num_features,
                causal=args.causal,
                attention_mode=mode,
                exact_threshold=args.exact_threshold,
                exact_backend=exact_backend,
                causal_chunk_size=args.causal_chunk_size,
                projection_seed=1,
            ).to(device=device, dtype=dtype)
            attention.eval()

            latency_ms, peak_mib = _measure(
                attention,
                inputs,
                mask,
                args.warmup,
                args.iterations,
            )
            tokens_per_second = args.batch_size * length * 1_000.0 / latency_ms
            rows.append(
                [
                    mode,
                    str(length),
                    f"{latency_ms:.3f}",
                    f"{tokens_per_second:,.0f}",
                    "n/a" if peak_mib is None else f"{peak_mib:.1f}",
                ]
            )
            del attention
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(
        f"device={device}, dtype={dtype}, batch={args.batch_size}, "
        f"exact_backend={exact_backend}"
    )
    print(_format_table(rows))


if __name__ == "__main__":
    main()
