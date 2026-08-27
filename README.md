<img src="./photo.png" width="500px"></img>

# Data-Aware Random Feature Kernels for Transformers (DARKformer)

[![PyPI version](https://badge.fury.io/py/darkformer-pytorch.svg)](https://badge.fury.io/py/darkformer-pytorch)

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
The additive feature floor is disabled by default; set `eps` to a small positive
value in linear mode when underflow protection is more important than the exact
estimator. Exact and automatic modes require `eps=0` so both paths evaluate the
same learned kernel.

The stabilized public feature maps rescale features by factors that cancel during
normalized attention. For direct kernel estimation with feature inner products,
use `stabilize=False` and `eps=0` to retain the unbiased Equation (3) estimator.

### Data-aware initialization

The geometry starts at the identity when no calibration data is available. Before
finetuning a pretrained model, it can instead be initialized from representative
queries and keys. This optional initializer is a library feature based on the
whitening construction in Proposition C.1. The paper does not specify covariance
whitening as the initialization used in its experiments.

The high-level attention initializers first apply the same $d_h^{-1/4}$ scaling
used by the kernel. They estimate the pooled within-query and within-key covariance
$\Lambda$ and set $M$ to a regularized symmetric $\Lambda^{-1/2}$. If queries and
keys have the same covariance, as assumed by Proposition C.1, this gives

```math
\mathrm{Cov}(Mq)=\mathrm{Cov}(Mk)=I.
```

Here $q$ and $k$ denote the scaled kernel inputs. If their empirical covariances
differ, the pooled estimate is a symmetric compromise and does not whiten both
distributions exactly.

Literal whitening does not preserve the pre-calibration attention temperature.
If raw projected queries have covariance $\Lambda_0$, unregularized calibration
sets

```math
M=d_h^{1/4}\Lambda_0^{-1/2},
```

so the calibrated score is $q_0^\mathsf{T}\Lambda_0^{-1}k_0$ rather than
$q_0^\mathsf{T}\Lambda_0^{-1}k_0/\sqrt{d_h}$. Leave the identity initialization
in place when a temperature-preserving start is more important than literal
Proposition C.1 whitening.

```python
import torch

from darkformer_pytorch import DarkformerLM

model = DarkformerLM(
    vocab_size=32_000,
    dim=512,
    depth=8,
    heads=8,
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
| `"auto"` | Uses exact attention through `exact_threshold`, then linear attention. |

For automatic selection, provide the cutoff explicitly:

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
redraws after that many training forwards. Evaluation forwards do not advance the
schedule.

All public attention and model modules expose the same in-place lifecycle methods:

```python
attention.redraw_projection_matrices_()
attention.fix_projection_matrices_()
attention.redraw_projection_matrices_(force=True)
attention.unfix_projection_matrices_()
```

Fixed projections ignore ordinary manual and scheduled redraws. Pass `force=True`
for an intentional one-time redraw while fixed. `projection_seed` makes initial
projections reproducible independently of PyTorch's global random state.
`deterministic=True` fixes projection matrices at construction. It also passes the
backend deterministic flag to FlashAttention 2 and 3, while the SDPA fallback uses
its math backend. Configure PyTorch's global deterministic settings separately when
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

## Benchmark

The synthetic microbenchmark compares PyTorch SDPA math, PyTorch's forced fused
FlashAttention backend, Performer, and DARKformer on the same held-out
anisotropic tensors. It measures kernel execution and does not reproduce the
paper's model finetuning experiments.

Performer and DARKformer use the same feature count, IID or orthogonal feature
structure, projection seeds, and additive feature floor. Performer projections
are injected explicitly instead of using its constructor defaults. DARKformer is
whitened from a separate calibration sample, and calibration is excluded from
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
Benchmark results will be added after the first reproducible GPU run.
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
