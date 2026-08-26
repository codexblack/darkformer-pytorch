# DARKformer

`darkformer` provides a PyTorch implementation of the data-aware random feature
kernel described in [Data-Aware Random Feature Kernel for
Transformers](https://arxiv.org/abs/2603.04127). It follows the positive random
feature formulation used by Performer while learning the projection geometry from
data.

## Kernel

For each attention head, DARKformer replaces the usual dot-product kernel with

```text
k(q, k) = exp(q^T Sigma k),       Sigma = M^T M.
```

The factorization keeps `Sigma` positive semidefinite. With `omega` sampled from a
standard Gaussian, the corresponding positive random feature map is

```text
phi(x; omega) = exp(omega^T M x - 0.5 ||M x||^2).
```

The finite feature map approximates the learned kernel, and normalized attention
can be evaluated associatively:

```text
phi(Q) (phi(K)^T V)
--------------------
phi(Q) (phi(K)^T 1)
```

This ordering does not construct the sequence-by-sequence score matrix. Its cost is
linear in sequence length and in the random feature count. Learning `M` aligns the
sampling covariance with the query-key geometry, which the paper interprets as an
implicit importance-sampling scheme for reducing Monte Carlo variance.

The learned positive semidefinite kernel and its positive random feature estimator
come from the paper. Runtime mode selection, feature count, redraw timing, exact
attention cutoff, language-model dimensions, and backend dispatch are library
configuration choices.

## Installation

Install the package from the repository root:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

PyTorch is the only runtime dependency. FlashAttention is optional and should be
installed separately for a compatible CUDA, PyTorch, and GPU environment.

## Attention

```python
import torch

from darkformer import DarkformerAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = DarkformerAttention(
    dim=512,
    heads=8,
    dim_head=64,
    num_features=256,
    attention_mode="linear",
    causal=True,
).to(device)

x = torch.randn(2, 2048, 512, device=device)
mask = torch.ones(2, 2048, dtype=torch.bool, device=device)
output = attention(x, mask=mask)
```

The input and output shapes are `(batch, sequence, dim)`. A boolean `mask` has shape
`(batch, sequence)`, where `True` marks a valid token. Causal attention can be set on
the module and combines with the token mask.

### Attention modes

`attention_mode` controls how the learned kernel is evaluated:

| Mode | Behavior |
| --- | --- |
| `"linear"` | Uses the positive random feature estimator and associative linear attention. |
| `"exact"` | Evaluates the learned kernel with exact softmax attention. |
| `"auto"` | Uses exact attention for short inputs and linear attention above the configured cutoff. |

The exact path applies the learned geometry to queries and keys before scaled
dot-product attention. It attempts FlashAttention 3, then FlashAttention 2, when an
installed backend supports the device, dtype, head dimension, causality, and mask.
It otherwise uses PyTorch scaled dot-product attention. Optional FlashAttention
packages are never required to import or run `darkformer`.

FlashAttention only serves the exact learned-kernel path. The linear positive random
feature path has no softmax score matrix for a FlashAttention kernel to compute.

### Random feature redraw

Projection redraw is explicit. This avoids changing the estimator as a side effect
of a forward pass:

```python
attention.redraw_projection_matrix()
```

Call redraw at a training boundary chosen by the application. Evaluation remains
deterministic until the next explicit redraw.

### Mixed precision

Move the module to CUDA and choose a model dtype in the usual PyTorch manner:

```python
attention = attention.to("cuda", dtype=torch.bfloat16)
x = x.to("cuda", dtype=torch.bfloat16)

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = attention(x, mask=mask.to("cuda"))
```

`bfloat16` is generally preferable where supported because of its wider exponent
range. Numerically sensitive normalization and reduction operations use stable
accumulation before results are returned in the model dtype.

## Language model

`DarkformerLM` composes causal DARKformer blocks into a decoder-only language model:

```python
import torch

from darkformer import DarkformerLM

model = DarkformerLM(
    num_tokens=32_000,
    dim=512,
    depth=8,
    heads=8,
    dim_head=64,
    num_features=256,
    max_seq_len=4096,
    attention_mode="linear",
).to("cuda")

tokens = torch.randint(0, 32_000, (2, 1024), device="cuda")
mask = torch.ones_like(tokens, dtype=torch.bool)
logits = model(tokens, mask=mask)

model.redraw_projection_matrices()
```

The returned logits have shape `(batch, sequence, num_tokens)`.

## Benchmark

The benchmark compares `linear`, `exact`, and `auto` modes using the same public
attention API. It does not import an optional FlashAttention package directly, so
the exact mode remains available through PyTorch on any supported installation.

```bash
python benchmarks/benchmark_attention.py --device cuda --dtype bfloat16
```

Run the benchmark with `--help` to configure sequence lengths, feature count, model
dimensions, warmup, and measurement iterations.

## References

- Amirhossein Farzam, Hossein Mobahi, Nolan Andrew Miller, and Luke Sernau.
  [Data-Aware Random Feature Kernel for
  Transformers](https://arxiv.org/abs/2603.04127), 2026.
- The [performer-pytorch](https://github.com/lucidrains/performer-pytorch)
  implementation is a useful reference for positive random feature attention and
  projection redraw behavior.
