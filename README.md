<img src="./photo.png" width="500px"></img>

# Data-Aware Random Feature Kernels for Transformers (DARKformer)

[![PyPI version](https://flat.badgen.net/pypi/v/darkformer-pytorch)](https://pypi.org/project/darkformer-pytorch/)

The `darkformer-pytorch` package provides a PyTorch implementation of the
data-aware random feature kernel described in the [Data-Aware Random Feature
Kernel for Transformers](https://arxiv.org/abs/2603.04127) paper by Google
DeepMind. It follows the positive random feature formulation used by Performer
while learning the projection geometry from data.

The public API follows conventions from lucidrains'
[performer-pytorch](https://github.com/lucidrains/performer-pytorch) package.

## Kernel

For each attention head, DARKformer replaces the usual dot-product kernel with:
<br></br>
```math
\begin{aligned}
\Sigma &= M^\mathsf{T} M \succeq 0, \\
\kappa_\Sigma(q, k) &= \exp\!\left(q^\mathsf{T} \Sigma k\right).
\end{aligned}
```
<br></br>
The public attention modules apply $d_h^{-1/4}$ to both queries and keys. For
unscaled inputs, the evaluated kernel is therefore
<br></br>
```math
\kappa_\Sigma(q, k)
= \exp\!\left(\frac{q^\mathsf{T}\Sigma k}{\sqrt{d_h}}\right).
```
<br></br>
The factorization keeps $\Sigma$ positive semidefinite. For $m$ features with each
$\omega_j$ sampled from a standard Gaussian, the corresponding positive random
feature map is

```math
\phi_\Sigma(x; \omega_j)
= \frac{1}{\sqrt{m}}
  \exp\!\left(
    \omega_j^\mathsf{T} Mx
    - \frac{1}{2} x^\mathsf{T} \Sigma x
  \right),
\qquad
\omega_j \sim \mathcal{N}(0, I).
```
<br></br>
The finite feature map approximates the learned kernel, and normalized attention
can be evaluated associatively:
<br></br>

```math
\mathrm{Att}(Q, K, V)
\approx
\frac{
  \Phi(Q)\left(\Phi(K)^\mathsf{T} V\right)
}{
  \Phi(Q)\left(\Phi(K)^\mathsf{T} \mathbf{1}\right)
}.
```
<br></br>
For sequence length $L$, head dimension $d_h$, and $m$ random features, this ordering
costs $O(L m d_h)$ per head and does not construct the $L \times L$ score matrix.
Exact attention costs $O(L^2 d_h)$. Learning $M$ aligns the sampling covariance with
the query-key geometry, which the paper interprets as an implicit
importance-sampling scheme for reducing Monte Carlo variance.

The learned positive semidefinite kernel and its positive random feature estimator
come from the paper. Runtime mode selection, feature count, redraw timing, exact
attention cutoff, per-head geometry, low-rank geometry, orthogonal feature blocks,
model depth, and backend dispatch are configurable library choices.

## Installation

Install from PyPI:

```bash
python -m pip install darkformer-pytorch
```

For development, install from the repository root:

```bash
python -m pip install -e ".[dev]"
```

PyTorch 2.4 or newer is the only runtime dependency.
[FlashAttention](https://github.com/Dao-AILab/flash-attention) is optional and
should be installed separately for a compatible CUDA, PyTorch, and GPU
environment.

## Self-attention

`DarkformerAttention` is the primary attention API and an alias of
`SelfAttention`.

```python
import torch

from darkformer_pytorch import DarkformerAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = DarkformerAttention(
    dim=512,
    heads=8,
    head_dim=64,
    num_features=256,
    geometry_rank=64,
    attention_mode="linear",
    causal=True,
).to(device)

x = torch.randn(2, 2048, 512, device=device)
mask = torch.ones(2, 2048, dtype=torch.bool, device=device)
output = attention(x, mask=mask)
```

The input and output shapes are $B \times L \times d$. A boolean `mask` has shape
$B \times L$, where `True` marks a valid token. Causal attention combines the token
mask with the causal constraint.

Independent Gaussian features are the default, matching Equation (3). Set
`orthogonal_features=True` to use Performer-style orthogonal Gaussian blocks.
The additive feature floor is disabled by default. Linear attention treats a
normalization mass at or below `torch.finfo(dtype).tiny` as an underflowed row and
returns a zero output with zero gradients, preventing a finite forward pass from
creating NaNs in division backward. Set `eps` to a small positive value in linear
mode only when avoiding these zero fallback rows is more important than retaining
the paper's estimator. Exact and automatic modes reject `eps > 0` because the
floor would introduce an additional mismatch between their exact and finite-feature
paths.

The stabilized public feature maps rescale features by factors that cancel during
normalized attention. For direct kernel estimation with feature inner products,
use `stabilize=False` and `eps=0` to retain the unbiased Equation (3) estimator.
With the default `eps=0`, stabilization subtracts the actual maximum even when it
is negative. This keeps an all-negative logit vector representable instead of
letting every exponential underflow to zero. When `eps > 0`, the shift is clamped
at zero because the rescaled correction `eps * exp(-shift)` could otherwise
overflow. The same policy is used by noncausal features and causal running state.

### Data-aware initialization

The geometry starts at the identity when no calibration data is available. Before
finetuning a pretrained model, it can instead be initialized from representative
queries and keys. This optional initializer is a library feature based on the
whitening construction in Proposition C.1. The paper does not specify covariance
whitening as the initialization used in its experiments.

The high-level attention initializers first apply the same $d_h^{-1/4}$ scaling
used by the kernel. They estimate the pooled within-query and within-key covariance
$\Lambda$ and set $M$ to a regularized symmetric inverse root with an explicit
post-whitening scale $s$:

```math
M=s\Lambda^{-1/2},\qquad
\mathrm{Cov}(Mq)=\mathrm{Cov}(Mk)=s^2I.
```

Here $q$ and $k$ denote the scaled kernel inputs. If their empirical covariances
differ, the pooled estimate is a symmetric compromise and does not whiten both
distributions exactly. Pass $s$ as `geometry_scale`. The default remains `1.0`
for backward compatibility and literal Proposition C.1 whitening.

For raw projected queries with covariance $\Lambda_0$, unregularized calibration
sets

```math
M=s\,d_h^{1/4}\Lambda_0^{-1/2}.
```

The calibrated score is therefore
$s^2q_0^\mathsf{T}\Lambda_0^{-1}k_0$, while the expected squared transformed norm
is approximately $d_hs^2$. Useful scale policies are:

| Policy | `geometry_scale` | Expected $\lVert Mq\rVert^2$ | Effect |
| :--- | :--- | :--- | :--- |
| Literal Proposition C.1 | `1.0` | $d_h$ | Unit covariance; highest dynamic range |
| Temperature preserving | `head_dim**-0.25` | $\sqrt{d_h}$ | Restores the usual $1/\sqrt{d_h}$ Mahalanobis score scale |
| Unit expected norm | `head_dim**-0.5` | $1$ | Lower feature variance, but colder attention |

The variance of exponential random features grows rapidly with the transformed
norm, so literal whitening can be impractical at ordinary head dimensions.
`initialize_whitening_` emits a warning reporting the selected scale and expected
norm. Covariance `shrinkage` is different: it regularizes the estimated covariance
toward an isotropic shape but does not uniformly reduce $M$.

Proposition C.1's whitening geometry is also distinct from Theorem 3.2's
variance-optimal proposal
$\Sigma^*=(I+2\Lambda)(I-2\Lambda)^{-1}$. DARKformer couples kernel geometry and
sampling covariance through the same learned matrix, so `initialize_whitening_`
does not implement that separate proposal. `geometry_scale` rescales the coupled
geometry; it is not a substitute for $\Sigma^*$. Validate any calibrated geometry
with the feature count and data distribution used in training.

```python
import torch

from darkformer_pytorch import DarkformerLM

model = DarkformerLM(
    vocab_size=32_000,
    dim=512,
    depth=8,
    heads=8,
    head_dim=64,
    num_features=256,
    max_seq_len=4096,
).to("cuda")

calibration_tokens = torch.randint(
    0,
    32_000,
    (8, 1024),
    device="cuda",
)

model.initialize_whitening_(
    calibration_tokens,
    regularization=1e-4,
    shrinkage=0.01,
    geometry_scale=64**-0.25,
)
```

`SelfAttention.initialize_whitening_(inputs, mask=...)` and
`CrossAttention.initialize_whitening_(inputs, context, ...)` provide the same
calibration for standalone modules. `DarkformerKernelAttention` accepts already
projected, unscaled tensors with shape $B \times H \times L \times d_h$ and applies
the kernel scaling internally. `DataAwareRandomFeatures.initialize_whitening_`
instead whitens the tensors passed directly to it without applying attention
scaling.

Full-rank geometry is required for whitening and for the density-ratio argument in
Proposition 4.1. Setting `geometry_rank < head_dim` produces a singular covariance;
the kernel estimator remains valid, but the full-density importance-sampling
interpretation does not. Construction emits a warning for that configuration.
Full configured rank is necessary but does not guarantee $\Sigma \succ 0$ throughout
training because $M$ is unconstrained and can become singular. Set
`per_head_geometry=False` to estimate one covariance shared by every head. The
default estimates each head separately. Per-head geometry is a library extension;
the paper's derivation treats one query-key distribution.

### Attention modes

`attention_mode` controls how the learned kernel is evaluated:

| Mode | Behavior |
| --- | --- |
| `"linear"` | Uses positive random features and associative linear attention. |
| `"exact"` | Evaluates the learned kernel with exact softmax attention. |
| `"auto"` | Uses exact attention through `exact_threshold`, then switches to the finite-feature approximation. |

Exact and linear modes compute materially different functions at finite feature
counts. Consequently, `"auto"` is an explicit accuracy/performance policy, not a
backend-only optimization: output can change discontinuously when a sequence
crosses the cutoff. `exact_threshold` is required with `attention_mode="auto"` and
is independent of `num_features`.

Provide the cutoff explicitly:

```python
attention = DarkformerAttention(
    512,
    heads=8,
    attention_mode="auto",
    exact_threshold=1024,
    exact_backend="auto",
).to("cuda")
```

The exact path applies the learned geometry to queries and keys before scaled
dot-product attention. `exact_backend="auto"` attempts FlashAttention 3, then
FlashAttention 2, when an installed backend supports the device, dtype, head
dimension, dropout, causality, and mask. It otherwise uses PyTorch scaled dot-product
attention. Set `exact_backend` to `"flash3"`, `"flash2"`, or `"sdpa"` to request a
specific backend. A forced FlashAttention backend raises an error when its package or
required hardware support is unavailable.

FlashAttention 3 requires an NVIDIA Hopper GPU and CUDA 12.3 or newer.
FlashAttention 2 requires CUDA 12.0 or newer on supported NVIDIA GPUs, or a
supported ROCm environment.

Optional FlashAttention packages are never required to import or run
`darkformer_pytorch`.
FlashAttention only serves the exact learned-kernel path. The linear positive random
feature path has no softmax score matrix for a FlashAttention kernel to compute.

## Cross-attention

`CrossAttention` keeps query and context masks separate. It has no causal or rotary
option because position handling belongs to the surrounding encoder-decoder model.

```python
import torch

from darkformer_pytorch import CrossAttention

cross_attention = CrossAttention(
    dim=512,
    heads=8,
    head_dim=64,
    num_features=256,
    attention_mode="linear",
).to("cuda")

x = torch.randn(2, 256, 512, device="cuda")
context = torch.randn(2, 1024, 512, device="cuda")
mask = torch.ones(2, 256, dtype=torch.bool, device="cuda")
context_mask = torch.ones(2, 1024, dtype=torch.bool, device="cuda")

output = cross_attention(
    x,
    context,
    mask=mask,
    context_mask=context_mask,
)
```

## Projection lifecycle

Random projections stay unchanged unless a redraw is requested. The default
`feature_redraw_interval=None` disables scheduled redraws. A positive interval
redraws after that many training forwards that belong to a linear-capable module.
In `"auto"` mode, short forwards routed through exact attention still advance the
schedule, so the interval counts training steps rather than only linear-path
executions. Exact-only modules do not redraw unused random features. Evaluation
forwards do not advance the schedule.

All public attention and model modules expose the same in-place lifecycle methods:

```python
attention.redraw_projection_matrices_()
attention.fix_projection_matrices_()
attention.redraw_projection_matrices_(force=True)
attention.unfix_projection_matrices_()
```

Fixed projections ignore ordinary manual and scheduled redraws. Pass `force=True`
for an intentional one-time redraw while fixed. `projection_seed` makes initial
projections reproducible independently of PyTorch's global random state. Use
`fixed_projection=True` to fix projection matrices at construction, and use
`backend_deterministic=True` to request deterministic exact-backend behavior from
FlashAttention 2 or 3 and the SDPA math fallback. These controls are independent.
The historical `deterministic` argument remains as a compatibility shorthand that
sets both policies; either explicit policy argument overrides its corresponding
legacy value. Configure PyTorch's global deterministic settings separately when
end-to-end determinism is required.

For a scheduled training policy:

```python
from darkformer_pytorch import SelfAttention

attention = SelfAttention(
    512,
    num_features=256,
    feature_redraw_interval=1_000,
    projection_seed=7,
)
```

## Mixed precision

Move the module to CUDA and select a model dtype with standard PyTorch operations:

```python
attention = attention.to("cuda", dtype=torch.bfloat16)
x = x.to("cuda", dtype=torch.bfloat16)
mask = mask.to("cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    output = attention(x, mask=mask)
```

We generally prefer `bfloat16` where supported because of its wider exponent
range. Numerically sensitive feature normalization and reductions use stable
accumulation before results are returned in the model dtype.

## Transformer stacks

`Darkformer` applies DARKformer attention and feed-forward layers to continuous
embeddings. Use `cross_attend=True` to add a context-attention sublayer.

```python
import torch

from darkformer_pytorch import Darkformer

encoder = Darkformer(
    dim=512,
    depth=8,
    heads=8,
    num_features=256,
    causal=False,
).to("cuda")

decoder = Darkformer(
    dim=512,
    depth=8,
    heads=8,
    num_features=256,
    causal=True,
    cross_attend=True,
).to("cuda")

source = torch.randn(2, 1024, 512, device="cuda")
target = torch.randn(2, 256, 512, device="cuda")
source_mask = torch.ones(2, 1024, dtype=torch.bool, device="cuda")
target_mask = torch.ones(2, 256, dtype=torch.bool, device="cuda")

context = encoder(source, mask=source_mask)
output = decoder(
    target,
    mask=target_mask,
    context=context,
    context_mask=source_mask,
)
```

## Language model

`DarkformerLM` composes causal DARKformer blocks into a decoder-only language model:

```python
import torch

from darkformer_pytorch import DarkformerLM

model = DarkformerLM(
    vocab_size=32_000,
    dim=512,
    depth=8,
    heads=8,
    head_dim=64,
    num_features=256,
    max_seq_len=4096,
    attention_mode="linear",
).to("cuda")

tokens = torch.randint(0, 32_000, (2, 1024), device="cuda")
mask = torch.ones_like(tokens, dtype=torch.bool)
logits = model(tokens, mask=mask)

model.redraw_projection_matrices_()
```

The returned logits have shape $B \times L \times V$, where $V$ is `vocab_size`.
`max_seq_len` is an optional input validation limit. `DarkformerLM` uses rotary
position information rather than learned absolute position embeddings by default.

## Encoder-decoder model and generation

`DarkformerEncDec` builds an encoder, a causal decoder, token embeddings, and output
projection. `encoder_depth` and `decoder_depth` can override the common `depth`.

```python
import torch

from darkformer_pytorch import DarkformerEncDec

model = DarkformerEncDec(
    source_vocab_size=32_000,
    target_vocab_size=32_000,
    dim=512,
    depth=8,
    heads=8,
    num_features=256,
    max_source_length=4096,
    max_target_length=1024,
    attention_mode="linear",
).to("cuda")

source_tokens = torch.randint(0, 32_000, (2, 1024), device="cuda")
target_tokens = torch.randint(0, 32_000, (2, 256), device="cuda")
source_mask = torch.ones_like(source_tokens, dtype=torch.bool)
target_mask = torch.ones_like(target_tokens, dtype=torch.bool)

logits = model(
    source_tokens,
    target_tokens,
    source_mask=source_mask,
    target_mask=target_mask,
)
loss = model(
    source_tokens,
    target_tokens,
    source_mask=source_mask,
    target_mask=target_mask,
    labels=target_tokens,
)
```

Generate autoregressively from a target prompt:

```python
prompt = target_tokens[:, :1]
generated = model.generate(
    source_tokens,
    prompt,
    max_new_tokens=128,
    source_mask=source_mask,
    eos_token_id=2,
    temperature=0.8,
    top_k=50,
)
```

`top_k` keeps exactly that many candidates per batch item. If logits tie at the
cutoff, PyTorch's `topk` ordering chooses which tied entries remain; ties do not
increase the candidate count.

With `attention_mode="linear"`, generation processes the prompt once and then
updates recurrent self-attention statistics for each appended token. Decoder
cross-attention projects and summarizes the encoded source once per layer.
`"auto"` and `"exact"` modes retain full-prefix decoding because their exact path
requires a conventional key-value cache.

The recurrent APIs are also available directly through
`forward_with_state(...)` on `SelfAttention`, `Darkformer`, and `DarkformerLM`,
and through `decode_with_state(...)` on `DarkformerEncDec`. Cached states are
append-only and tied to the model parameters, device, dtype, masks, and random
projection matrices used to create them. Discard a state after changing any of
those inputs. A projection redraw is detected and rejected automatically.
The projection version and redraw counter are checkpointed, so stale cached states
remain rejectable after saving and reloading a model.

## Benchmark

The synthetic microbenchmark compares PyTorch SDPA math, PyTorch's forced fused
FlashAttention backend, Performer, and DARKformer on the same held-out
anisotropic tensors. It measures kernel execution and does not reproduce the
paper's model finetuning experiments.

Performer and DARKformer use the same feature count, IID or orthogonal feature
structure, projection seeds, and additive feature floor. Performer projections
are injected explicitly instead of using its constructor defaults. DARKformer is
whitened from a separate calibration sample using the configurable
`--geometry-scale` (literal `1.0` by default), and calibration is excluded from
timed regions. Performance rows report the median and IQR across repeated blocked
timings. GPU memory is the incremental peak allocation during one warmed forward,
not total process or model memory.

Approximation error is measured in float32 over 30 projection seeds by default.
Performer is compared with isotropic SDPA math. DARKformer is compared with exact
Mahalanobis attention using the same calibrated geometry. These rows measure the
finite-feature error against each method's target kernel; they are not a direct
comparison of model quality.

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/benchmark_attention.py --device cuda --dtype bfloat16
```

Raw measurements are written under the ignored `benchmark-results/` directory.
The JSON records the method order, sequence lengths, error and calibration sizes,
precision policy, feature controls, Git revision and dirty state, PyTorch and
package versions, CUDA version, GPU, and NVIDIA driver. Only formatted tables and
their exact configuration are committed here.

<!-- benchmark-table:start -->
Results from commit `e6202ec` on 2026-08-27 are shown below. The system used an
NVIDIA GeForce RTX 4070 Ti (compute capability 8.9), driver 610.47, CUDA 13.0,
PyTorch 2.12.0+cu130, Python 3.12.13, `performer-pytorch` 1.1.4, and
`darkformer-pytorch` 0.1.1.

The noncausal workload used batch size 1, 8 heads, head dimension 64, 256 IID
features, `eps=0`, data seed 17, and projection seed 1,000. Query and key inputs
had covariance condition number 16. DARKformer used a disjoint length-512
calibration sample with regularization $10^{-4}$, shrinkage $0.01$, and literal
`geometry_scale=1.0`.
Performance inputs used bfloat16. Performer kept bfloat16 features and
reductions; DARKformer used bfloat16 projections with float32 features and
reductions. Each latency is the median of five blocked timing repeats after
three warmups; the value in parentheses is the IQR. Every timing repeat ran for
at least 0.25 seconds. Memory is the incremental peak CUDA allocation during one
warmed forward.

| Sequence | Method | Median latency, ms (IQR) | Tokens/s | Peak MiB |
| ---: | :--- | ---: | ---: | ---: |
| 512 | SDPA math | 0.163 (0.001) | 3,139,938 | 22.0 |
| 512 | Performer | 0.623 (0.014) | 822,079 | 8.0 |
| 512 | DARKformer | 0.730 (0.007) | 701,570 | 16.0 |
| 1,024 | SDPA math | 0.512 (0.001) | 2,000,426 | 80.0 |
| 1,024 | Performer | 0.620 (0.012) | 1,652,909 | 24.0 |
| 1,024 | DARKformer | 0.738 (0.002) | 1,387,526 | 34.0 |
| 2,048 | SDPA math | 2.919 (0.018) | 701,627 | 304.0 |
| 2,048 | Performer | 0.596 (0.009) | 3,434,997 | 48.0 |
| 2,048 | DARKformer | 0.695 (0.008) | 2,946,966 | 68.0 |
| 4,096 | SDPA math | 11.363 (0.015) | 360,481 | 1,184.0 |
| 4,096 | Performer | 0.585 (0.014) | 7,007,519 | 78.0 |
| 4,096 | DARKformer | 1.967 (0.070) | 2,082,359 | 128.0 |

The forced PyTorch fused-SDPA method was unavailable because this PyTorch build
was not compiled with FlashAttention. No fallback value is reported for that
method. On this workload, DARKformer crossed math SDPA between 1,024 and 2,048
tokens. At 4,096 tokens, its median latency was 5.78 times lower and its
incremental peak allocation was 9.25 times lower than math SDPA.

Approximation error used separate float32 tensors of length 512 and a separate
calibration sample. Results cover 30 projection seeds starting at 1,000; values
are median relative L2 error with IQR in parentheses.

| Method | Reference | Relative L2 error (IQR) |
| :--- | :--- | ---: |
| Performer | Isotropic softmax, SDPA math | 2.507232 (0.229741) |
| DARKformer | Exact held-out calibrated Mahalanobis attention | 1.364384 (0.007816) |

The error rows use different target kernels and are not directly comparable.
They measure random-feature approximation error, not downstream model quality.
<!-- benchmark-table:end -->

## References
```bibtex
 @misc{farzam2026dataawarerandomfeaturekernel,
      title={Data-Aware Random Feature Kernel for Transformers}, 
      author={Amirhossein Farzam and Hossein Mobahi and Nolan Andrew Miller and Luke Sernau},
      year={2026},
      eprint={2603.04127},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.04127}, 
}
```
```bibtex
@misc{choromanski2020rethinking,
    title   = {Rethinking Attention with Performers},
    author  = {Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
    year    = {2020},
    eprint  = {2009.14794},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
