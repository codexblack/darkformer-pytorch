"""Tests for exact attention backends."""

import math

import pytest
import torch

from darkformer import backends


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if mask is not None:
        allowed = mask[:, None, None, :].expand_as(scores)
    else:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        length = query.shape[2]
        allowed = allowed & torch.ones(
            (length, length), dtype=torch.bool, device=query.device
        ).tril()
    scores = scores.masked_fill(~allowed, -torch.inf)
    output = torch.softmax(scores, dim=-1) @ value
    if mask is not None:
        output = output.masked_fill(~mask[:, None, :, None], 0.0)
    return output


def test_sdpa_matches_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 3, 5, 4, generator=generator, dtype=torch.float64)
    key = torch.randn(2, 3, 5, 4, generator=generator, dtype=torch.float64)
    value = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.float64)

    actual = backends.exact_attention(
        query,
        key,
        value,
        causal=False,
        mask=None,
        dropout_p=0.0,
        backend="sdpa",
        scale=0.3,
    )
    expected = _reference_attention(
        query, key, value, causal=False, mask=None, scale=0.3
    )

    torch.testing.assert_close(actual, expected)


def test_sdpa_combines_padding_and_causal_masks() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, 2, 4, 3, generator=generator)
    key = torch.randn(2, 2, 4, 3, generator=generator)
    value = torch.randn(2, 2, 4, 5, generator=generator)
    mask = torch.tensor(
        [[True, False, True, True], [True, True, False, False]]
    )

    actual = backends.exact_attention(
        query,
        key,
        value,
        causal=True,
        mask=mask,
        dropout_p=0.0,
        backend="sdpa",
        scale=1.0 / math.sqrt(3),
    )
    expected = _reference_attention(
        query,
        key,
        value,
        causal=True,
        mask=mask,
        scale=1.0 / math.sqrt(3),
    )

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[~mask[:, None, :].expand(-1, 2, -1)]) == 0


def test_auto_prefers_flash3_and_converts_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    calls = []

    def fake_flash3(
        flash_query: torch.Tensor,
        flash_key: torch.Tensor,
        flash_value: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        calls.append(
            (
                flash_query.shape,
                flash_key.shape,
                flash_value.shape,
                softmax_scale,
                causal,
            )
        )
        return flash_value

    monkeypatch.setattr(
        backends, "_flash3_unavailable_reason", lambda *args: None
    )
    monkeypatch.setattr(backends, "_load_flash3", lambda: (fake_flash3, None))
    monkeypatch.setattr(
        backends,
        "_load_flash2",
        lambda: pytest.fail("FlashAttention 2 should not be loaded"),
    )

    output = backends.exact_attention(
        query,
        key,
        value,
        causal=True,
        mask=None,
        dropout_p=0.0,
        scale=0.25,
    )

    assert calls == [
        (torch.Size([2, 5, 3, 4]),) * 3 + (0.25, True)
    ]
    torch.testing.assert_close(output, value)


def test_auto_skips_flash3_dropout_and_uses_flash2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    received = {}

    def fake_flash2(
        flash_query: torch.Tensor,
        flash_key: torch.Tensor,
        flash_value: torch.Tensor,
        *,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        received.update(
            shape=flash_query.shape,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        return flash_value

    monkeypatch.setattr(
        backends,
        "_flash3_unavailable_reason",
        lambda *args: "attention dropout is unsupported",
    )
    monkeypatch.setattr(
        backends, "_flash2_unavailable_reason", lambda *args: None
    )
    monkeypatch.setattr(backends, "_load_flash2", lambda: (fake_flash2, None))

    output = backends.exact_attention(
        query,
        key,
        value,
        causal=False,
        mask=None,
        dropout_p=0.2,
        scale=0.5,
    )

    assert received == {
        "shape": torch.Size([1, 3, 2, 4]),
        "dropout_p": 0.2,
        "softmax_scale": 0.5,
        "causal": False,
    }
    torch.testing.assert_close(output, value)


@pytest.mark.parametrize("backend", ["flash3", "flash2"])
def test_forced_flash_backend_reports_ineligible_device(backend: str) -> None:
    tensor = torch.randn(1, 1, 2, 4)

    with pytest.raises(RuntimeError, match=f"FlashAttention {backend[-1]} is unavailable"):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            mask=None,
            dropout_p=0.0,
            backend=backend,
        )


def test_forced_flash_backend_reports_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 2, 4)
    monkeypatch.setattr(
        backends, "_flash3_unavailable_reason", lambda *args: None
    )
    monkeypatch.setattr(
        backends,
        "_load_flash3",
        lambda: (None, OSError("binary is incompatible")),
    )

    with pytest.raises(RuntimeError, match="OSError: binary is incompatible"):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            mask=None,
            dropout_p=0.0,
            backend="flash3",
        )


def test_rejects_invalid_padding_mask() -> None:
    tensor = torch.randn(2, 1, 3, 4)

    with pytest.raises(TypeError, match="torch.bool"):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            mask=torch.ones(2, 3),
            dropout_p=0.0,
        )


def test_rejects_unknown_backend() -> None:
    tensor = torch.randn(1, 1, 2, 4)

    with pytest.raises(ValueError, match="unknown attention backend"):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            mask=None,
            dropout_p=0.0,
            backend="unknown",
        )
