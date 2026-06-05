<div align="center">

# Hunyuan3D-2.1 + HiCache++

**Tencent's Hunyuan3D-2.1 image/text-to-3D, accelerated by the HiCache++ *exponential* velocity cache on its DiT flow-matching loop.**

*HiCache++ replaces HiCache's polynomial forecast basis with a Dynamic-Mode-Decomposition (Prony) **exponential** basis — exact on the class diffusion features actually live in, so it stays lossless at larger skip intervals than the polynomial. HiCache (Hermite) is kept as the in-fork comparison baseline.*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![Hunyuan3D&#8209;2.1](https://img.shields.io/badge/base-Hunyuan3D--2.1-d96902)
&nbsp;![arXiv](https://img.shields.io/badge/HiCache-arXiv%3A2508.16984-b5212f?logo=arxiv)

</div>

---

## What this is

[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (© Tencent) generates a textured 3D asset from a single image (or text): a DiT **flow-matching** sampler denoises a latent shape, then a paint stage textures it. The shape sampler is the cost — one DiT forward per sampling step.

This fork accelerates that loop with **HiCache++**. On most sampling steps the expensive `self.model(...)` forward is skipped and the (CFG-combined) flow-matching velocity is *forecast* from cached anchors, so the DiT runs only every `interval` steps. The forecaster is a **first-class part of the pipeline** — `Hunyuan3DDiTFlowMatchingPipeline.__call__` reads it natively, with **no monkey-patching** — and is training-free and geometry-preserving. The Hermite (polynomial) HiCache forecaster ships too, as the comparison baseline.

## Method — DMD exponential forecast

Every modern feature cache skips the network and *forecasts* the velocity; they differ in the **basis** used to extrapolate, and the basis is what sets the skip ceiling. HiCache / TaylorSeer / FoCa forecast with a **polynomial** (or rational) basis. But a diffusion feature trajectory across timesteps is the solution of a near-linear feature-ODE, whose **exact** solution class is a sum of (damped / oscillatory) **exponentials** — `F_t = Σ_j a_j e^{μ_j t}`. A polynomial is only a local Taylor truncation of that, so it **diverges** under extrapolation, which is exactly why every polynomial cache caps out at a modest skip.

HiCache++ forecasts with **Dynamic Mode Decomposition** (Schmid 2010) — the SVD-regularised generalisation of **Prony's method** (1795) / **Matrix-Pencil** (Hua–Sarkar 1990). It identifies the linear propagator `A` from raw velocity snapshots (`F_{t+1} ≈ A F_t`), eigendecomposes it once, and predicts any (fractional) horizon `k` by eigenvalue powers:

```
F_{t+k} ≈ Φ (λ**k ⊙ b),     b = Φ⁺ F_t
```

Because this **is** the exact solution class — not a truncation of it — DMD is exact on exponential trajectories and holds quality at skip intervals where the Hermite/Taylor polynomial drifts. One economy SVD of a `[d, n]` snapshot matrix (`d ≫ n`) is cheap relative to a DiT forward. DMD needs a **≥4-snapshot uniform window** (a real-valued conjugate pole pair costs two real DOF, so even one oscillatory mode needs 3 snapshot-pairs); during warm-up or across the non-uniform first-enhance boundary, it falls back to the Hermite forecast, so DMD acts only where it is valid.

## How to enable it

```python
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

# Exponential (DMD/Prony) forecaster — the new method:
pipe.enable_dmd(
    interval=5,        # one DiT forward, then interval-1 DMD forecasts
    first_enhance=2,   # always compute the first few steps (warm-up)
    history=6,         # snapshot window the DMD fit is built from
)

# equivalently, via the unified HiCache entry point with the exponential backend:
pipe.enable_hicache(interval=5, backend="dmd", history=6)

# baseline for comparison — the Hermite polynomial forecaster (default backend):
# pipe.enable_hicache(interval=3, backend="hermite", sigma=0.5)

mesh = pipe(image="assets/demo.png")[0]   # same call as upstream — caching is transparent
pipe.disable_hicache()
```

Both `enable_dmd` and `enable_hicache(backend="dmd")` just record the schedule; the denoise loop in [`hy3dshape/hy3dshape/pipelines.py`](hy3dshape/hy3dshape/pipelines.py) dispatches skipped steps to `dmd_forecast_state` (exponential) or `hicache_forecast` (Hermite fallback) by the `backend` flag. The exponential forecaster lives in [`hy3dshape/hy3dshape/hicache_dmd.py`](hy3dshape/hy3dshape/hicache_dmd.py); the Hermite baseline in [`hy3dshape/hy3dshape/hicache.py`](hy3dshape/hy3dshape/hicache.py). Both are CPU-testable with no GPU or model: `python -m hy3dshape.hy3dshape.hicache_dmd`.

## Results

At interval-5, HiCache++ (DMD) holds **F-score ≈ 0.83** where HiCache (Hermite) drops to **≈ 0.74** (uncached baseline ≈ 0.89) — Toys4K F-score@0.05, 3-seed, excluding one Go-ICP-degenerate sphere. **DMD's lead over Hermite grows with the skip interval**: the exponential basis is what extends the lossless skip range past the point where the polynomial collapses.

For the full cross-model benchmarks (controlled forecast microbenchmark, Hunyuan3D-2.1, Hunyuan3D-2-mini, SAM3D, Fast-SAM3D) and the complete Hermite-vs-exponential tables, see the standalone library **[`hicache-plus-plus`](../hicache-plus-plus)**.

## Attribution

- **Base model:** [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) © Tencent — see [`PROJECT.md`](PROJECT.md) and the upstream license. All Hunyuan3D-2.1 code, weights, and trademarks belong to Tencent.
- **HiCache** (the polynomial baseline): *HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting* (arXiv:[2508.16984](https://arxiv.org/abs/2508.16984)). Reimplemented here as the comparison backend.
- **HiCache++** (this work): the **DMD/Prony exponential** forecaster. DMD (Schmid 2010) / Prony (1795) / Matrix-Pencil (Hua–Sarkar 1990) are classical spectral estimation; their application to **diffusion feature caching** is, to our knowledge, new.

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
