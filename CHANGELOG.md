# Changelog

## 1.0.0 - 2026-08-30

This is the first stable release of `darkformer-pytorch`.

### Highlights

- Corrected the QR sign ambiguity in orthogonal Gaussian projections, restoring
  unbiased positive random-feature estimates while retaining their variance
  reduction.
- Made orthogonal features the default. Independent Gaussian projections remain
  available with `orthogonal_features=False`.
- Reworked causal stabilization to compute scale boundaries on-device without
  per-chunk `.nonzero()` or `.item()` synchronization, and changed the default
  causal chunk size from 64 to 256.
- Kept stabilized FP16/BF16 feature activations, reductions, and recurrent state in
  the model dtype while retaining FP32 logits and stabilization scales.
- Added the variance-optimal proposal from Theorem 3.2, including exact importance
  weights, validation of the normalizability condition, reset APIs, low-rank
  support, and checkpoint coverage.
- Changed whitening initialization to use the temperature-preserving
  `head_dim**-0.25` geometry scale by default. Literal unit-covariance whitening
  remains available with `geometry_scale=1.0`.
- Mixed the persistent redraw counter into each module's projection seed to prevent
  projection-stream collisions when only a subset of layers is fixed.
- Hardened optional FlashAttention dispatch for both supported FlashAttention 3
  import layouts and tuple returns, and added explicit head-dimension eligibility
  checks.
- Removed device synchronizations caused by inspecting all-true key masks. Pass
  `key_mask=None` when every key is valid and maskless FlashAttention dispatch is
  desired.

### Compatibility notes

- Existing state dictionaries continue to load, including state dictionaries from
  before projection-version tracking was introduced.
- Saved projection matrices are unchanged when loaded. Newly constructed modules
  now use orthogonal projections by default, and future seeded redraw streams use
  the collision-resistant seed mixer.
- `attention_mode="auto"` still requires an explicit `exact_threshold`; the README
  now documents why the exact-to-linear crossover must be benchmarked for the
  target feature count and hardware.
