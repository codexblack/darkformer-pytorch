"""Transformer stacks built with DARKformer attention."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from darkformer.attention import (
    AttentionMode,
    CrossAttention,
    SelfAttention,
)
from darkformer.backends import AttentionBackend


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transformation."""
        gate, values = self.input_projection(inputs).chunk(2, dim=-1)
        hidden = functional.silu(gate) * values
        return self.dropout(self.output_projection(hidden))


class DarkformerBlock(nn.Module):
    """Pre-normalized DARKformer block."""

    self_attention: SelfAttention
    cross_attention: CrossAttention | None

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        head_dim: int | None,
        num_features: int | None,
        geometry_rank: int | None,
        mlp_dim: int,
        causal: bool,
        cross_attend: bool,
        attention_mode: AttentionMode,
        exact_threshold: int | None,
        exact_backend: AttentionBackend,
        dropout: float,
        rotary: bool,
        rotary_base: float,
        causal_chunk_size: int,
        feature_redraw_interval: int | None,
        projection_seed: int | None,
        deterministic: bool,
    ) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(dim)
        self.self_attention = SelfAttention(
            dim,
            heads=heads,
            head_dim=head_dim,
            num_features=num_features,
            geometry_rank=geometry_rank,
            causal=causal,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            dropout=dropout,
            rotary=rotary,
            rotary_base=rotary_base,
            causal_chunk_size=causal_chunk_size,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        self.cross_norm = nn.RMSNorm(dim) if cross_attend else None
        self.cross_attention = (
            CrossAttention(
                dim,
                heads=heads,
                head_dim=head_dim,
                num_features=num_features,
                geometry_rank=geometry_rank,
                attention_mode=attention_mode,
                exact_threshold=exact_threshold,
                exact_backend=exact_backend,
                dropout=dropout,
                causal_chunk_size=causal_chunk_size,
                feature_redraw_interval=feature_redraw_interval,
                projection_seed=(
                    None if projection_seed is None else projection_seed + 1
                ),
                deterministic=deterministic,
            )
            if cross_attend
            else None
        )
        self.feed_forward_norm = nn.RMSNorm(dim)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention, optional cross-attention, and feed-forward layers."""
        inputs = inputs + self.self_attention(self.self_norm(inputs), mask=mask)
        if self.cross_attention is not None:
            if context is None or self.cross_norm is None:
                raise ValueError("context is required when cross_attend=True")
            inputs = inputs + self.cross_attention(
                self.cross_norm(inputs),
                context,
                mask=mask,
                context_mask=context_mask,
            )
        inputs = inputs + self.feed_forward(self.feed_forward_norm(inputs))
        return inputs


class Darkformer(nn.Module):
    """A stack of DARKformer blocks for encoded input sequences."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        heads: int = 8,
        head_dim: int | None = None,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        mlp_dim: int | None = None,
        causal: bool = False,
        cross_attend: bool = False,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        dropout: float = 0.0,
        rotary: bool = True,
        rotary_base: float = 10_000.0,
        causal_chunk_size: int = 64,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1 or depth < 1:
            raise ValueError("dim and depth must be positive")
        mlp_dim = 4 * dim if mlp_dim is None else mlp_dim
        if mlp_dim < 1:
            raise ValueError("mlp_dim must be positive")
        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.cross_attend = cross_attend
        layers = []
        for index in range(depth):
            layer_seed = (
                None if projection_seed is None else projection_seed + 2 * index
            )
            layers.append(
                DarkformerBlock(
                    dim,
                    heads=heads,
                    head_dim=head_dim,
                    num_features=num_features,
                    geometry_rank=geometry_rank,
                    mlp_dim=mlp_dim,
                    causal=causal,
                    cross_attend=cross_attend,
                    attention_mode=attention_mode,
                    exact_threshold=exact_threshold,
                    exact_backend=exact_backend,
                    dropout=dropout,
                    rotary=rotary,
                    rotary_base=rotary_base,
                    causal_chunk_size=causal_chunk_size,
                    feature_redraw_interval=feature_redraw_interval,
                    projection_seed=layer_seed,
                    deterministic=deterministic,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _attention_modules(self) -> Iterator[SelfAttention | CrossAttention]:
        for module in self.layers:
            layer = cast(DarkformerBlock, module)
            yield layer.self_attention
            if layer.cross_attention is not None:
                yield layer.cross_attention

    def redraw_projection_matrices_(
        self,
        *,
        force: bool = False,
    ) -> Darkformer:
        """Redraw every random-feature basis in place."""
        for attention in self._attention_modules():
            attention.redraw_projection_matrices_(force=force)
        return self

    def fix_projection_matrices_(self) -> Darkformer:
        """Disable projection redraws in every layer."""
        for attention in self._attention_modules():
            attention.fix_projection_matrices_()
        return self

    def unfix_projection_matrices_(self) -> Darkformer:
        """Enable projection redraws in every layer."""
        for attention in self._attention_modules():
            attention.unfix_projection_matrices_()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform a batch-first sequence."""
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"inputs must have shape [batch, length, {self.dim}]")
        if self.cross_attend and context is None:
            raise ValueError("context is required when cross_attend=True")
        for layer in self.layers:
            inputs = layer(
                inputs,
                mask=mask,
                context=context,
                context_mask=context_mask,
            )
        return inputs


class DarkformerLM(nn.Module):
    """Token language model backed by a DARKformer stack."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        *,
        heads: int = 8,
        head_dim: int | None = None,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        mlp_dim: int | None = None,
        max_seq_len: int | None = None,
        causal: bool = True,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        rotary: bool = True,
        rotary_base: float = 10_000.0,
        causal_chunk_size: int = 64,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if max_seq_len is not None and max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.transformer = Darkformer(
            dim,
            depth,
            heads=heads,
            head_dim=head_dim,
            num_features=num_features,
            geometry_rank=geometry_rank,
            mlp_dim=mlp_dim,
            causal=causal,
            cross_attend=False,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            dropout=dropout,
            rotary=rotary,
            rotary_base=rotary_base,
            causal_chunk_size=causal_chunk_size,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        self.final_norm = nn.RMSNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def _validate_tokens(self, tokens: torch.Tensor, name: str) -> None:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer tensor [batch, length]")
        if self.max_seq_len is not None and tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"{name} length {tokens.shape[1]} exceeds {self.max_seq_len}"
            )

    def redraw_projection_matrices_(
        self,
        *,
        force: bool = False,
    ) -> DarkformerLM:
        """Redraw every random-feature basis in place."""
        self.transformer.redraw_projection_matrices_(force=force)
        return self

    def fix_projection_matrices_(self) -> DarkformerLM:
        """Disable projection redraws in every layer."""
        self.transformer.fix_projection_matrices_()
        return self

    def unfix_projection_matrices_(self) -> DarkformerLM:
        """Enable projection redraws in every layer."""
        self.transformer.unfix_projection_matrices_()
        return self

    def forward_features(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized token representations."""
        self._validate_tokens(tokens, "tokens")
        hidden = self.token_embedding(tokens)
        hidden = self.transformer(hidden, mask=mask)
        return self.final_norm(hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits or next-token cross-entropy when labels are provided."""
        logits = self.output_projection(self.forward_features(tokens, mask=mask))
        if labels is None:
            return logits
        self._validate_tokens(labels, "labels")
        if labels.shape != tokens.shape:
            raise ValueError("labels and tokens must have the same shape")
        if tokens.shape[1] < 2:
            raise ValueError("at least two tokens are required for language-model loss")
        return functional.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend token prompts."""
        self._validate_tokens(prompt, "prompt")
        if not self.causal:
            raise RuntimeError("generation requires a causal language model")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        generated = prompt
        finished = torch.zeros(
            prompt.shape[0],
            dtype=torch.bool,
            device=prompt.device,
        )
        try:
            for _ in range(max_new_tokens):
                logits = self(generated)[:, -1] / temperature
                if top_k is not None:
                    count = min(top_k, logits.shape[-1])
                    threshold = logits.topk(count, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
                if eos_token_id is not None:
                    eos = torch.full_like(next_token, eos_token_id)
                    next_token = torch.where(finished[:, None], eos, next_token)
                    finished |= next_token.squeeze(1).eq(eos_token_id)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return generated


class DarkformerEncDec(nn.Module):
    """Encoder-decoder transformer with DARKformer self and cross attention."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        dim: int,
        depth: int,
        *,
        encoder_depth: int | None = None,
        decoder_depth: int | None = None,
        heads: int = 8,
        head_dim: int | None = None,
        num_features: int | None = None,
        geometry_rank: int | None = None,
        mlp_dim: int | None = None,
        max_source_length: int | None = None,
        max_target_length: int | None = None,
        attention_mode: AttentionMode = "linear",
        exact_threshold: int | None = None,
        exact_backend: AttentionBackend = "auto",
        dropout: float = 0.0,
        tie_embeddings: bool = False,
        rotary: bool = True,
        rotary_base: float = 10_000.0,
        causal_chunk_size: int = 64,
        feature_redraw_interval: int | None = None,
        projection_seed: int | None = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if min(source_vocab_size, target_vocab_size) < 1:
            raise ValueError("vocabulary sizes must be positive")
        if tie_embeddings and source_vocab_size != target_vocab_size:
            raise ValueError("tied embeddings require equal vocabulary sizes")
        encoder_depth = depth if encoder_depth is None else encoder_depth
        decoder_depth = depth if decoder_depth is None else decoder_depth
        if min(depth, encoder_depth, decoder_depth) < 1:
            raise ValueError("depth values must be positive")
        for name, length in (
            ("max_source_length", max_source_length),
            ("max_target_length", max_target_length),
        ):
            if length is not None and length < 1:
                raise ValueError(f"{name} must be positive")
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_embedding = nn.Embedding(source_vocab_size, dim)
        self.target_embedding = nn.Embedding(target_vocab_size, dim)
        nn.init.normal_(self.source_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.target_embedding.weight, mean=0.0, std=0.02)
        if tie_embeddings:
            self.target_embedding.weight = self.source_embedding.weight
        self.encoder = Darkformer(
            dim,
            encoder_depth,
            heads=heads,
            head_dim=head_dim,
            num_features=num_features,
            geometry_rank=geometry_rank,
            mlp_dim=mlp_dim,
            causal=False,
            cross_attend=False,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            dropout=dropout,
            rotary=rotary,
            rotary_base=rotary_base,
            causal_chunk_size=causal_chunk_size,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=projection_seed,
            deterministic=deterministic,
        )
        decoder_seed = (
            None if projection_seed is None else projection_seed + 2 * encoder_depth
        )
        self.decoder = Darkformer(
            dim,
            decoder_depth,
            heads=heads,
            head_dim=head_dim,
            num_features=num_features,
            geometry_rank=geometry_rank,
            mlp_dim=mlp_dim,
            causal=True,
            cross_attend=True,
            attention_mode=attention_mode,
            exact_threshold=exact_threshold,
            exact_backend=exact_backend,
            dropout=dropout,
            rotary=rotary,
            rotary_base=rotary_base,
            causal_chunk_size=causal_chunk_size,
            feature_redraw_interval=feature_redraw_interval,
            projection_seed=decoder_seed,
            deterministic=deterministic,
        )
        self.encoder_norm = nn.RMSNorm(dim)
        self.decoder_norm = nn.RMSNorm(dim)
        self.output_projection = nn.Linear(dim, target_vocab_size, bias=False)
        if tie_embeddings:
            self.output_projection.weight = self.target_embedding.weight

    def _validate_tokens(
        self,
        tokens: torch.Tensor,
        *,
        name: str,
        max_length: int | None,
    ) -> None:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be an integer tensor [batch, length]")
        if max_length is not None and tokens.shape[1] > max_length:
            raise ValueError(f"{name} length {tokens.shape[1]} exceeds {max_length}")

    def redraw_projection_matrices_(
        self,
        *,
        force: bool = False,
    ) -> DarkformerEncDec:
        """Redraw every encoder and decoder projection basis."""
        self.encoder.redraw_projection_matrices_(force=force)
        self.decoder.redraw_projection_matrices_(force=force)
        return self

    def fix_projection_matrices_(self) -> DarkformerEncDec:
        """Disable all encoder and decoder projection redraws."""
        self.encoder.fix_projection_matrices_()
        self.decoder.fix_projection_matrices_()
        return self

    def unfix_projection_matrices_(self) -> DarkformerEncDec:
        """Enable all encoder and decoder projection redraws."""
        self.encoder.unfix_projection_matrices_()
        self.decoder.unfix_projection_matrices_()
        return self

    def encode(
        self,
        source_tokens: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source tokens once for decoding or generation."""
        self._validate_tokens(
            source_tokens,
            name="source_tokens",
            max_length=self.max_source_length,
        )
        encoded = self.encoder(
            self.source_embedding(source_tokens),
            mask=source_mask,
        )
        return self.encoder_norm(encoded)

    def decode(
        self,
        target_tokens: torch.Tensor,
        context: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target tokens against encoded source context."""
        self._validate_tokens(
            target_tokens,
            name="target_tokens",
            max_length=self.max_target_length,
        )
        decoded = self.decoder(
            self.target_embedding(target_tokens),
            mask=target_mask,
            context=context,
            context_mask=source_mask,
        )
        return self.output_projection(self.decoder_norm(decoded))

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return decoder logits or next-token cross-entropy loss."""
        if source_tokens.shape[0] != target_tokens.shape[0]:
            raise ValueError("source and target batch sizes must match")
        context = self.encode(source_tokens, source_mask=source_mask)
        logits = self.decode(
            target_tokens,
            context,
            target_mask=target_mask,
            source_mask=source_mask,
        )
        if labels is None:
            return logits
        self._validate_tokens(
            labels,
            name="labels",
            max_length=self.max_target_length,
        )
        if labels.shape != target_tokens.shape:
            raise ValueError("labels and target_tokens must have the same shape")
        if target_tokens.shape[1] < 2:
            raise ValueError("at least two target tokens are required for loss")
        return functional.cross_entropy(
            logits[:, :-1].reshape(-1, self.target_vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        source_tokens: torch.Tensor,
        prompt: torch.Tensor,
        *,
        max_new_tokens: int,
        source_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate target tokens while reusing the encoded source context."""
        if source_tokens.shape[0] != prompt.shape[0]:
            raise ValueError("source and prompt batch sizes must match")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        generated = prompt
        finished = torch.zeros(
            prompt.shape[0],
            dtype=torch.bool,
            device=prompt.device,
        )
        try:
            context = self.encode(source_tokens, source_mask=source_mask)
            for _ in range(max_new_tokens):
                logits = self.decode(
                    generated,
                    context,
                    source_mask=source_mask,
                )[:, -1]
                logits = logits / temperature
                if top_k is not None:
                    count = min(top_k, logits.shape[-1])
                    threshold = logits.topk(count, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
                if eos_token_id is not None:
                    eos = torch.full_like(next_token, eos_token_id)
                    next_token = torch.where(finished[:, None], eos, next_token)
                    finished |= next_token.squeeze(1).eq(eos_token_id)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
        finally:
            self.train(was_training)
        return generated


__all__ = [
    "Darkformer",
    "DarkformerBlock",
    "DarkformerEncDec",
    "DarkformerLM",
]
