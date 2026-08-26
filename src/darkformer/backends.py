"""Exact attention backend selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F


AttentionBackend = Literal["auto", "flash3", "flash2", "sdpa"]
_FlashAttention = Callable[..., torch.Tensor]
_FLASH_DTYPES = (torch.float16, torch.bfloat16)


def _load_flash3() -> tuple[_FlashAttention | None, BaseException | None]:
    try:
        from flash_attn_3 import flash_attn_interface
    except (ImportError, OSError) as error:
        return None, error
    return flash_attn_interface.flash_attn_func, None


def _load_flash2() -> tuple[_FlashAttention | None, BaseException | None]:
    try:
        from flash_attn import flash_attn_func
    except (ImportError, OSError) as error:
        return None, error
    return flash_attn_func, None


def _flash3_unavailable_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    dropout_p: float,
) -> str | None:
    if query.device.type != "cuda" or torch.version.hip is not None:
        return "a CUDA device with compute capability 9.0 is required"
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
    if mask is not None:
        return "padding masks are unsupported"
    if dropout_p != 0.0:
        return "attention dropout is unsupported"
    return None


def _flash2_unavailable_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> str | None:
    if query.device.type != "cuda" or torch.version.hip is not None:
        return "an NVIDIA Ampere, Ada, or Hopper GPU is required"
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
    if mask is not None:
        return "padding masks are unsupported"
    return None


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    dropout_p: float,
    backend: str,
    scale: float,
) -> None:
    if backend not in ("auto", "flash3", "flash2", "sdpa"):
        raise ValueError(f"unknown attention backend: {backend!r}")
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, heads, length, dim]")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if query.shape[:3] != key.shape[:3] or query.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value batch, head, and length dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if mask is not None:
        expected_shape = (query.shape[0], query.shape[2])
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if mask.device != query.device:
            raise ValueError("mask must be on the same device as query")
        if tuple(mask.shape) != expected_shape:
            raise ValueError(f"mask must have shape {expected_shape}")
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
) -> torch.Tensor:
    output = function(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
    )
    return output.transpose(1, 2)


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    mask: torch.Tensor | None,
    dropout_p: float,
    scale: float,
) -> torch.Tensor:
    attention_mask = None
    use_causal_flag = causal
    if mask is not None:
        attention_mask = mask[:, None, None, :]
        if causal:
            length = query.shape[2]
            causal_mask = torch.ones(
                (length, length), dtype=torch.bool, device=query.device
            ).tril_()
            attention_mask = attention_mask & causal_mask
            use_causal_flag = False
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=use_causal_flag,
        scale=scale,
    )
    if mask is not None:
        output = output.masked_fill(~mask[:, None, :, None], 0.0)
    return output


def _unavailable_error(
    name: str,
    reason: str | None,
    import_error: BaseException | None = None,
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
    mask: torch.Tensor | None,
    dropout_p: float,
    backend: AttentionBackend = "auto",
    scale: float = 1.0,
) -> torch.Tensor:
    """Computes exact scaled dot-product attention.

    Args:
      query: Query tensor with shape `[batch, heads, length, head_dim]`.
      key: Key tensor with shape `[batch, heads, length, head_dim]`.
      value: Value tensor with shape `[batch, heads, length, value_dim]`.
      causal: Whether to restrict attention to current and previous positions.
      mask: Boolean tensor with shape `[batch, length]`; true values are valid.
      dropout_p: Attention dropout probability.
      backend: Attention implementation to use.
      scale: Scale applied to query-key scores.

    Returns:
      The attention output with the same leading dimensions as `query`.

    Raises:
      RuntimeError: A requested FlashAttention backend is unavailable.
      TypeError: An input has an invalid type or dtype.
      ValueError: An input has an invalid shape or value.
    """
    dropout_p = float(dropout_p)
    scale = float(scale)
    _validate_inputs(query, key, value, mask, dropout_p, backend, scale)

    flash3_reason = _flash3_unavailable_reason(
        query, key, value, mask, dropout_p
    )
    if backend in ("auto", "flash3") and flash3_reason is None:
        flash3, import_error = _load_flash3()
        if flash3 is not None:
            return _run_flash3(flash3, query, key, value, causal, scale)
        if backend == "flash3":
            raise _unavailable_error("FlashAttention 3", None, import_error)
    elif backend == "flash3":
        raise _unavailable_error("FlashAttention 3", flash3_reason)

    flash2_reason = _flash2_unavailable_reason(query, key, value, mask)
    if backend in ("auto", "flash2") and flash2_reason is None:
        flash2, import_error = _load_flash2()
        if flash2 is not None:
            return _run_flash2(
                flash2,
                query,
                key,
                value,
                causal,
                dropout_p,
                scale,
            )
        if backend == "flash2":
            raise _unavailable_error("FlashAttention 2", None, import_error)
    elif backend == "flash2":
        raise _unavailable_error("FlashAttention 2", flash2_reason)

    return _run_sdpa(query, key, value, causal, mask, dropout_p, scale)
