"""Tests for exact attention backends."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
import torch

from darkformer_pytorch import backends


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
    query_mask: torch.Tensor | None,
    key_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if key_mask is not None:
        allowed = key_mask[:, None, None, :].expand_as(scores)
    else:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if causal:
        length = query.shape[2]
        allowed = (
            allowed
            & torch.ones((length, length), dtype=torch.bool, device=query.device).tril()
        )
    scores = scores.masked_fill(~allowed, -torch.inf)
    output = torch.softmax(scores, dim=-1) @ value
    if query_mask is not None:
        output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
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
        dropout_p=0.0,
        backend="sdpa",
        scale=0.3,
    )
    expected = _reference_attention(
        query,
        key,
        value,
        causal=False,
        query_mask=None,
        key_mask=None,
        scale=0.3,
    )

    torch.testing.assert_close(actual, expected)


def test_sdpa_combines_padding_and_causal_masks() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, 2, 4, 3, generator=generator)
    key = torch.randn(2, 2, 4, 3, generator=generator)
    value = torch.randn(2, 2, 4, 5, generator=generator)
    query_mask = torch.tensor([[True, False, True, True], [True, True, False, False]])
    key_mask = query_mask.clone()

    actual = backends.exact_attention(
        query,
        key,
        value,
        causal=True,
        query_mask=query_mask,
        key_mask=key_mask,
        dropout_p=0.0,
        backend="sdpa",
        scale=1.0 / math.sqrt(3),
    )
    expected = _reference_attention(
        query,
        key,
        value,
        causal=True,
        query_mask=query_mask,
        key_mask=key_mask,
        scale=1.0 / math.sqrt(3),
    )

    torch.testing.assert_close(actual, expected)
    invalid_queries = ~query_mask[:, None, :].expand(-1, 2, -1)
    assert torch.count_nonzero(actual[invalid_queries]) == 0


def test_sdpa_supports_masked_cross_attention() -> None:
    generator = torch.Generator().manual_seed(13)
    query = torch.randn(2, 3, 2, 4, generator=generator)
    key = torch.randn(2, 3, 5, 4, generator=generator)
    value = torch.randn(2, 3, 5, 6, generator=generator)
    query_mask = torch.tensor([[True, False], [True, True]])
    key_mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )

    actual = backends.exact_attention(
        query,
        key,
        value,
        causal=False,
        query_mask=query_mask,
        key_mask=key_mask,
        dropout_p=0.0,
        backend="sdpa",
        scale=0.5,
    )
    expected = _reference_attention(
        query,
        key,
        value,
        causal=False,
        query_mask=query_mask,
        key_mask=key_mask,
        scale=0.5,
    )

    torch.testing.assert_close(actual, expected)


def test_sdpa_returns_zero_when_every_key_is_masked() -> None:
    query = torch.randn(2, 2, 3, 4)
    key = torch.randn(2, 2, 5, 4)
    value = torch.randn(2, 2, 5, 6)
    key_mask = torch.tensor(
        [[False, False, False, False, False], [True, False, True, False, True]]
    )

    output = backends.exact_attention(
        query,
        key,
        value,
        causal=False,
        key_mask=key_mask,
        dropout_p=0.0,
        backend="sdpa",
    )

    assert torch.all(torch.isfinite(output))
    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(output[1]) > 0


def test_sdpa_deterministic_uses_math_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 3, 4)
    events: list[str] = []

    @contextmanager
    def fake_math_context() -> Iterator[None]:
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(backends, "_sdpa_math_context", fake_math_context)

    backends.exact_attention(
        tensor,
        tensor,
        tensor,
        causal=False,
        dropout_p=0.0,
        backend="sdpa",
        deterministic=True,
    )

    assert events == ["enter", "exit"]


def test_auto_uses_sdpa_for_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 3, 4)
    query_mask = torch.tensor([[True, True, False]])
    key_mask = torch.tensor([[True, False, True]])
    monkeypatch.setattr(
        backends,
        "_load_flash3",
        lambda: pytest.fail("FlashAttention 3 should not be loaded"),
    )
    monkeypatch.setattr(
        backends,
        "_load_flash2",
        lambda: pytest.fail("FlashAttention 2 should not be loaded"),
    )

    output = backends.exact_attention(
        tensor,
        tensor,
        tensor,
        causal=False,
        query_mask=query_mask,
        key_mask=key_mask,
        dropout_p=0.0,
    )

    assert torch.count_nonzero(output[:, :, -1]) == 0


def test_auto_prefers_flash3_and_converts_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    calls: list[
        tuple[
            torch.Size,
            torch.Size,
            torch.Size,
            float,
            bool,
            bool,
        ]
    ] = []

    def fake_flash3(
        flash_query: torch.Tensor,
        flash_key: torch.Tensor,
        flash_value: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        calls.append(
            (
                flash_query.shape,
                flash_key.shape,
                flash_value.shape,
                softmax_scale,
                causal,
                deterministic,
            )
        )
        return flash_value

    monkeypatch.setattr(backends, "_flash3_unavailable_reason", lambda *args: None)
    monkeypatch.setattr(backends, "_load_flash3", lambda: (fake_flash3, None))
    monkeypatch.setattr(
        backends,
        "_load_flash2",
        lambda: pytest.fail("FlashAttention 2 should not be loaded"),
    )

    query_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    output = backends.exact_attention(
        query,
        key,
        value,
        causal=True,
        query_mask=query_mask,
        dropout_p=0.0,
        scale=0.25,
    )

    assert calls == [
        (
            torch.Size([2, 5, 3, 4]),
            torch.Size([2, 5, 3, 4]),
            torch.Size([2, 5, 3, 4]),
            0.25,
            True,
            False,
        ),
    ]
    expected = value.masked_fill(~query_mask[:, None, :, None], 0.0)
    torch.testing.assert_close(output, expected)


def test_auto_ignores_an_all_true_key_mask_for_flash_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    key_mask = torch.ones(1, 3, dtype=torch.bool)
    received_masks: list[torch.Tensor | None] = []

    def availability(
        _query: torch.Tensor,
        _key: torch.Tensor,
        _value: torch.Tensor,
        _query_mask: torch.Tensor | None,
        dispatched_key_mask: torch.Tensor | None,
        _dropout_p: float,
    ) -> None:
        received_masks.append(dispatched_key_mask)

    def fake_flash3(
        _query: torch.Tensor,
        _key: torch.Tensor,
        flash_value: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        del softmax_scale, causal, deterministic
        return flash_value

    monkeypatch.setattr(backends, "_flash3_unavailable_reason", availability)
    monkeypatch.setattr(backends, "_load_flash3", lambda: (fake_flash3, None))

    output = backends.exact_attention(
        query,
        key,
        value,
        causal=False,
        key_mask=key_mask,
        dropout_p=0.0,
    )

    assert received_masks == [None]
    torch.testing.assert_close(output, value)


@pytest.mark.parametrize(
    ("version", "minimum", "expected"),
    [
        ("11.8", (12, 0), False),
        ("12.0", (12, 0), True),
        ("12.2", (12, 3), False),
        ("12.3", (12, 3), True),
        ("13.0", (12, 3), True),
        (None, (12, 0), False),
    ],
)
def test_cuda_version_requirement(
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
    minimum: tuple[int, int],
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", version)

    assert backends._cuda_version_at_least(*minimum) is expected


def test_auto_passes_deterministic_to_flash3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    received: dict[str, object] = {}

    def fake_flash3(
        flash_query: torch.Tensor,
        flash_key: torch.Tensor,
        flash_value: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        received.update(
            shape=flash_query.shape,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        return flash_value

    monkeypatch.setattr(backends, "_flash3_unavailable_reason", lambda *args: None)
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
        causal=False,
        dropout_p=0.0,
        scale=0.5,
        deterministic=True,
    )

    assert received == {
        "shape": torch.Size([1, 3, 2, 4]),
        "softmax_scale": 0.5,
        "causal": False,
        "deterministic": True,
    }
    torch.testing.assert_close(output, value)


@pytest.mark.parametrize("backend", ["flash3", "flash2"])
def test_forced_flash_backend_reports_ineligible_device(
    backend: backends.AttentionBackend,
) -> None:
    tensor = torch.randn(1, 1, 2, 4)

    message = f"FlashAttention {backend[-1]} is unavailable"
    with pytest.raises(RuntimeError, match=message):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            dropout_p=0.0,
            backend=backend,
        )


def test_forced_flash_backend_reports_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 2, 4)
    monkeypatch.setattr(backends, "_flash3_unavailable_reason", lambda *args: None)
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
            dropout_p=0.0,
            backend="flash3",
        )


def test_forced_flash3_passes_deterministic_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.randn(1, 1, 2, 4)
    received: list[bool] = []

    def fake_flash3(
        _query: torch.Tensor,
        _key: torch.Tensor,
        value: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        deterministic: bool,
    ) -> torch.Tensor:
        del softmax_scale, causal
        received.append(deterministic)
        return value

    monkeypatch.setattr(backends, "_flash3_unavailable_reason", lambda *args: None)
    monkeypatch.setattr(backends, "_load_flash3", lambda: (fake_flash3, None))

    output = backends.exact_attention(
        tensor,
        tensor,
        tensor,
        causal=False,
        dropout_p=0.0,
        backend="flash3",
        deterministic=True,
    )

    assert received == [True]
    torch.testing.assert_close(output, tensor)


def test_rejects_causal_cross_attention() -> None:
    query = torch.randn(1, 1, 2, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 5)

    with pytest.raises(ValueError, match="equal query and key lengths"):
        backends.exact_attention(
            query,
            key,
            value,
            causal=True,
            dropout_p=0.0,
        )


def test_rejects_invalid_padding_mask() -> None:
    tensor = torch.randn(2, 1, 3, 4)

    with pytest.raises(TypeError, match=r"torch\.bool"):
        backends.exact_attention(
            tensor,
            tensor,
            tensor,
            causal=False,
            query_mask=torch.ones(2, 3),
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
            dropout_p=0.0,
            backend=cast(backends.AttentionBackend, "unknown"),
        )
