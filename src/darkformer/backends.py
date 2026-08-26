"""Exact attention backend selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import lru_cache
from typing import Literal, cast

import torch
from torch.nn import functional

AttentionBackend = Literal["auto", "flash3", "flash2", "sdpa"]
_FlashAttention = Callable[..., torch.Tensor]
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _cuda_version_at_least(major: int, minor: int) -> bool:
    version = torch.version.cuda
    if version is None:
        return False
    try:
        installed_major, installed_minor = version.split(".")[:2]
        return (int(installed_major), int(installed_minor)) >= (major, minor)
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _load_flash3() -> tuple[_FlashAttention | None, Exception | None]:
    try:
        from flash_attn_3 import flash_attn_interface
    except (ImportError, OSError) as error:
        return None, error
    return cast(_FlashAttention, flash_attn_interface.flash_attn_func), None


@lru_cache(maxsize=1)
def _load_flash2() -> tuple[_FlashAttention | None, Exception | None]:
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as error:
        return None, error
    return cast(_FlashAttention, flash_attn_func), None


def _flash3_unavailable_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    dropout_p: float,
    deterministic: bool,
) -> str | None:
    if deterministic:
        return "deterministic execution is unsupported"
    if key_mask is not None:
        return "padding masks are unsupported"
    if dropout_p != 0.0:
        return "attention dropout is unsupported"
    if query.device.type != "cuda" or torch.version.hip is not None:
        return "a CUDA device with compute capability 9.0 is required"
    if not _cuda_version_at_least(12, 3):
        return "CUDA 12.3 or newer is required"
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        return "a CUDA device with compute capability 9.0 is required"
    if query.dtype not in _FLASH_DTYPES:
        return "FP16 or BF16 inputs are required"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "query, key, and value must have the same dtype"
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        return "query, key, and value head dimensions must match"
    if query.shape[-1] > 256:
        return "head dimensions greater than 256 are unsupported"
    return None


def _flash2_unavailable_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
) -> str | None:
    if query.device.type != "cuda":
        return "a supported CUDA or ROCm device is required"
    if torch.version.hip is None:
        if not _cuda_version_at_least(12, 0):
            return "CUDA 12.0 or newer is required"
        major, _ = torch.cuda.get_device_capability(query.device)
        if major not in (8, 9):
            return "an NVIDIA Ampere, Ada, or Hopper GPU is required"
    if query.dtype not in _FLASH_DTYPES:
        return "FP16 or BF16 inputs are required"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "query, key, and value must have the same dtype"
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        return "query, key, and value head dimensions must match"
    if query.shape[-1] > 256:
        return "head dimensions greater than 256 are unsupported"
    if key_mask is not None:
        return "padding masks are unsupported"
    return None


def _validate_mask(
    name: str,
    mask: torch.Tensor | None,
    expected_shape: tuple[int, int],
    device: torch.device,
) -> None:
    if mask is None:
        return
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != device:
        raise ValueError(f"{name} must be on the same device as query")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    dropout_p: float,
    backend: str,
    scale: float,
) -> None:
    if backend not in ("auto", "flash3", "flash2", "sdpa"):
        raise ValueError(f"unknown attention backend: {backend!r}")
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, heads, length, dim]")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value batch and head dimensions must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value length dimensions must match")
    if query.shape[2] < 1 or key.shape[2] < 1:
        raise ValueError("query and key lengths must be positive")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if causal and query.shape[2] != key.shape[2]:
        raise ValueError("causal attention requires equal query and key lengths")
    _validate_mask(
        "query_mask",
        query_mask,
        (query.shape[0], query.shape[2]),
        query.device,
    )
    _validate_mask(
        "key_mask",
        key_mask,
        (key.shape[0], key.shape[2]),
        query.device,
    )
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in the interval [0, 1)")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")


def _run_flash3(
    function: _FlashAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    output = function(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=scale,
        causal=causal,
    )
    return output.transpose(1, 2)


def _run_flash2(
    function: _FlashAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    dropout_p: float,
    scale: float,
    deterministic: bool,
) -> torch.Tensor:
    output = function(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        deterministic=deterministic,
    )
    return output.transpose(1, 2)


@contextmanager
def _legacy_sdpa_math_context() -> Iterator[None]:
    context_factory = getattr(torch.backends.cuda, "sdp_kernel", None)
    if context_factory is None:
        yield
        return
    try:
        context = context_factory(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
    except TypeError:
        context = context_factory(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    with context:
        yield


def _sdpa_math_context() -> AbstractContextManager[None]:
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        context = sdpa_kernel(SDPBackend.MATH)
    except (AttributeError, ImportError):
        return _legacy_sdpa_math_context()
    return cast(AbstractContextManager[None], context)


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    dropout_p: float,
    scale: float,
    deterministic: bool,
) -> torch.Tensor:
    attention_mask = None
    use_causal_flag = causal
    valid_key_rows = None
    if key_mask is not None:
        valid_key_rows = key_mask.any(dim=-1)
        first_key = torch.zeros_like(key_mask)
        first_key[:, 0] = True
        safe_key_mask = key_mask | (~valid_key_rows[:, None] & first_key)
        attention_mask = safe_key_mask[:, None, None, :]
        if causal:
            query_length = query.shape[2]
            key_length = key.shape[2]
            causal_mask = torch.ones(
                (query_length, key_length),
                dtype=torch.bool,
                device=query.device,
            ).tril_()
            attention_mask = attention_mask & causal_mask
            use_causal_flag = False
    context = _sdpa_math_context() if deterministic else nullcontext()
    with context:
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=use_causal_flag,
            scale=scale,
        )
    if query_mask is not None:
        output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
    if valid_key_rows is not None:
        output = output.masked_fill(~valid_key_rows[:, None, None, None], 0.0)
    return output


def _unavailable_error(
    name: str,
    reason: str | None,
    import_error: Exception | None = None,
) -> RuntimeError:
    if reason is not None:
        return RuntimeError(f"{name} is unavailable: {reason}")
    if import_error is not None:
        return RuntimeError(
            f"{name} is unavailable: {type(import_error).__name__}: {import_error}"
        )
    return RuntimeError(f"{name} is unavailable")


def exact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    dropout_p: float,
    backend: AttentionBackend = "auto",
    scale: float = 1.0,
    deterministic: bool = False,
) -> torch.Tensor:
    """Compute exact scaled dot-product attention.

    Args:
      query: Query tensor with shape `[batch, heads, query_length, head_dim]`.
      key: Key tensor with shape `[batch, heads, key_length, head_dim]`.
      value: Value tensor with shape `[batch, heads, key_length, value_dim]`.
      causal: Whether to restrict attention to current and previous positions.
      query_mask: Boolean tensor with shape `[batch, query_length]`; true values
        are valid.
      key_mask: Boolean tensor with shape `[batch, key_length]`; true values are
        valid.
      dropout_p: Attention dropout probability.
      backend: Attention implementation to use.
      scale: Scale applied to query-key scores.
      deterministic: Whether to select deterministic backend behavior.

    Returns:
      The attention output with the same leading dimensions as `query`.

    Raises:
      RuntimeError: A requested FlashAttention backend is unavailable.
      TypeError: An input has an invalid type or dtype.
      ValueError: An input has an invalid shape or value.
    """
    dropout_p = float(dropout_p)
    scale = float(scale)
    _validate_inputs(
        query,
        key,
        value,
        causal,
        query_mask,
        key_mask,
        dropout_p,
        backend,
        scale,
    )
    dispatch_key_mask = key_mask
    if key_mask is not None and bool(key_mask.all()):
        dispatch_key_mask = None

    if backend in ("auto", "flash3"):
        flash3_reason = _flash3_unavailable_reason(
            query,
            key,
            value,
            query_mask,
            dispatch_key_mask,
            dropout_p,
            deterministic,
        )
        if flash3_reason is None:
            flash3, import_error = _load_flash3()
            if flash3 is not None:
                output = _run_flash3(flash3, query, key, value, causal, scale)
                if query_mask is not None:
                    output = output.masked_fill(
                        ~query_mask[:, None, :, None],
                        0.0,
                    )
                return output
            if backend == "flash3":
                raise _unavailable_error("FlashAttention 3", None, import_error)
        elif backend == "flash3":
            raise _unavailable_error("FlashAttention 3", flash3_reason)

    if backend in ("auto", "flash2"):
        flash2_reason = _flash2_unavailable_reason(
            query,
            key,
            value,
            query_mask,
            dispatch_key_mask,
        )
        if flash2_reason is None:
            flash2, import_error = _load_flash2()
            if flash2 is not None:
                output = _run_flash2(
                    flash2,
                    query,
                    key,
                    value,
                    causal,
                    dropout_p,
                    scale,
                    deterministic,
                )
                if query_mask is not None:
                    output = output.masked_fill(
                        ~query_mask[:, None, :, None],
                        0.0,
                    )
                return output
            if backend == "flash2":
                raise _unavailable_error("FlashAttention 2", None, import_error)
        elif backend == "flash2":
            raise _unavailable_error("FlashAttention 2", flash2_reason)

    return _run_sdpa(
        query,
        key,
        value,
        causal,
        query_mask,
        dispatch_key_mask,
        dropout_p,
        scale,
        deterministic,
    )
