"""Adaptive-CFG (Adaptive Guidance) for Hunyuan3D-2.1.

Training-free CFG acceleration for the flow-matching denoise loop, ported from
faster-trellis. Adaptive Guidance (Castillo et al., arXiv:2312.12487) observes
that the conditional and unconditional velocity predictions become increasingly
*aligned* as sampling proceeds (cosine similarity gamma_t -> 1). Once
``gamma_t >= gamma_bar`` the unconditional pass carries little new directional
information, so we drop it and run a conditional-ONLY forward, reconstructing the
guidance term from cached anchors.

Hunyuan-specific CFG convention
-------------------------------
Hunyuan's loop combines the 2x-batched velocity as::

    v_cfg = v_uncond + w * (v_cond - v_uncond) = v_cond + (w - 1) * (v_cond - v_uncond)

so the cached/forecast **guidance term** is::

    g = (w - 1) * (v_cond - v_uncond),     v_cfg = v_cond + g                  (Hunyuan)

(TRELLIS uses ``v_cfg = (1+w)*v_cond - w*v_uncond`` -> ``g = w*(...)``; only this
coefficient differs.) At a *compute* step both halves of the 2x batch are run, g
is cached (anchored at the step index) and gamma measured; at a *skip* step only
the conditional half is run (latents at 1x batch + the conditional slice of the
stacked conditioning) and ``g`` is forecast, halving the model compute.

This is first-class pipeline code (the denoise loop calls these helpers directly);
there is NO runtime monkey-patching. Enable with ``pipe.enable_adaptive_guidance()``.
The Hermite/finite-difference math is model-agnostic; ``cond_first_half`` is the
only Hunyuan-specific helper (it slices the stacked conditioning to its
conditional half).
"""
from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# tensor helpers                                                              #
# --------------------------------------------------------------------------- #
def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """gamma_t: cosine similarity of the conditional and unconditional preds."""
    fa, fb = _flat(a), _flat(b)
    return float((torch.dot(fa, fb) / (fa.norm() * fb.norm() + eps)).item())


def cond_first_half(cond: Any, half: int) -> Any:
    """Conditional half of Hunyuan's stacked conditioning (``cat([cond, uncond])``).

    Recursively slices ``[:half]`` of every tensor in the (possibly nested-dict)
    conditioning so the model can run a conditional-only forward at 1x batch.
    Non-tensor / non-dict leaves pass through unchanged.
    """
    if isinstance(cond, torch.Tensor):
        return cond[:half]
    if isinstance(cond, dict):
        return {k: cond_first_half(v, half) for k, v in cond.items()}
    if isinstance(cond, (list, tuple)):
        return type(cond)(cond_first_half(v, half) for v in cond)
    return cond


# --------------------------------------------------------------------------- #
# guidance-term forecast (Newton divided differences)                          #
# --------------------------------------------------------------------------- #
def forecast_guidance(
    anchors: List[Tuple[int, torch.Tensor]],
    step: int,
    max_order: int = 1,
) -> torch.Tensor:
    """Forecast the guidance term ``g`` at integer ``step`` from cached anchors.

    Newton forward-difference extrapolation through the most recent
    ``max_order + 1`` anchors -- the unique polynomial of degree ``<= max_order``
    through them, hence *exact* for polynomial guidance series of that degree.
    Handles non-uniform step spacing (so it is robust to HiCache-skipped steps).
    Edge cases: 0 anchors -> ValueError; 1 anchor (or max_order<1) -> 0th-order
    hold (the vanilla Adaptive-Guidance behaviour).
    """
    if len(anchors) == 0:
        raise ValueError("forecast_guidance requires at least one anchor")
    if len(anchors) == 1 or max_order < 1:
        return anchors[-1][1].clone()

    used = anchors[-(max_order + 1):]
    xs = [float(s) for s, _ in used]
    ys = [g.clone() for _, g in used]
    n = len(used)

    coeffs = [ys[0]]
    col = ys
    for k in range(1, n):
        new_col = []
        for i in range(n - k):
            denom = xs[i + k] - xs[i]
            new_col.append((col[i + 1] - col[i]) / denom)
        col = new_col
        coeffs.append(col[0])

    x = float(step)
    result = coeffs[-1].clone()
    for k in range(n - 2, -1, -1):
        result = result * (x - xs[k]) + coeffs[k]
    return result


# --------------------------------------------------------------------------- #
# state + decision                                                            #
# --------------------------------------------------------------------------- #
def adaptive_cfg_init(
    num_steps: int,
    gamma_bar: float = 0.94,
    warmup: int = 2,
    max_order: int = 1,
) -> Dict[str, Any]:
    """Per-run Adaptive-CFG state.

    gamma_bar : cosine-sim threshold in [0,1]; higher -> more conservative.
    warmup    : force full CFG for the first ``warmup`` steps (build anchors).
    max_order : guidance-forecast polynomial order (1 = linear extrapolation).
    """
    if not (0.0 <= gamma_bar <= 1.0):
        raise ValueError(f"gamma_bar must be in [0,1], got {gamma_bar}")
    return {
        "num_steps": int(num_steps),
        "gamma_bar": float(gamma_bar),
        "warmup": int(warmup),
        "max_order": int(max_order),
        "step": 0,
        "anchors": [],          # list[(step, g)]
        "last_gamma": None,
        "n_full": 0,
        "n_skip": 0,
    }


def adaptive_cfg_decide(state: Dict[str, Any], gamma: Optional[float]) -> bool:
    """True if this step must run the FULL (uncond) CFG pass. A skip requires all
    of: past warmup, >=1 cached anchor, not the final step, last cosine >= bar."""
    step = state["step"]
    if step < state["warmup"]:
        return True
    if step >= state["num_steps"] - 1:          # always anchor the final step
        return True
    if len(state["anchors"]) == 0:
        return True
    if gamma is None:
        return True
    return gamma < state["gamma_bar"]


# --------------------------------------------------------------------------- #
# CPU unit test (no GPU, no model): validates the forecast + decision core      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        global ok
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    D = 32
    # 1) linear guidance series -> order-1 forecast exact
    A, B = torch.randn(D), torch.randn(D)
    g = lambda s: A + B * float(s)
    anchors = [(3, g(3)), (4, g(4))]
    for s in (5, 6, 7):
        check(f"linear forecast exact @ {s}",
              torch.allclose(forecast_guidance(anchors, s, 1), g(s), atol=1e-4))

    # 2) quadratic series -> order-2 forecast exact (incl. non-uniform anchors)
    A2, B2, C2 = torch.randn(D), torch.randn(D), torch.randn(D)
    q = lambda s: A2 + B2 * float(s) + C2 * float(s) ** 2
    anc2 = [(2, q(2)), (4, q(4)), (5, q(5))]   # non-uniform spacing (HiCache-skip robust)
    for s in (7, 9):
        check(f"quadratic forecast exact @ {s} (non-uniform)",
              torch.allclose(forecast_guidance(anc2, s, 2), q(s), atol=1e-2))

    # 3) edge cases
    check("single-anchor hold", torch.allclose(forecast_guidance([(5, g(5))], 99, 1), g(5)))
    try:
        forecast_guidance([], 0); check("zero-anchor raises", False)
    except ValueError:
        check("zero-anchor raises", True)

    # 4) cosine
    v = torch.randn(2, D)
    check("cosine self == 1", abs(cosine_sim(v, v) - 1.0) < 1e-5)
    check("cosine anti == -1", abs(cosine_sim(v, -v) + 1.0) < 1e-5)

    # 5) decisions
    st = adaptive_cfg_init(num_steps=10, gamma_bar=0.9, warmup=2, max_order=1)
    st["step"] = 0; check("warmup forces full", adaptive_cfg_decide(st, 0.99) is True)
    st["step"] = 3; check("no-anchor forces full", adaptive_cfg_decide(st, 0.99) is True)
    st["anchors"].append((2, torch.zeros(D)))
    check("aligned -> skip", adaptive_cfg_decide(st, 0.95) is False)
    check("misaligned -> full", adaptive_cfg_decide(st, 0.80) is True)
    st["step"] = 9; check("final step forces full", adaptive_cfg_decide(st, 0.99) is True)

    # 6) Hunyuan g convention: v_cfg = v_cond + g, g = (w-1)*(cond-uncond)
    w = 5.0
    vc, vu = torch.randn(D), torch.randn(D)
    g_hunyuan = (w - 1.0) * (vc - vu)
    v_cfg_direct = vu + w * (vc - vu)
    check("Hunyuan g: v_cond + g == uncond + w*(cond-uncond)",
          torch.allclose(vc + g_hunyuan, v_cfg_direct, atol=1e-5))

    # 7) cond_first_half slices a nested dict to the conditional half
    cc = {"main": torch.arange(8).reshape(8, 1).float(), "x": {"y": torch.arange(8).float()}}
    h = cond_first_half(cc, 4)
    check("cond_first_half slices nested dict to [:half]",
          h["main"].shape[0] == 4 and torch.allclose(h["x"]["y"], torch.arange(4).float()))

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    import sys
    sys.exit(0 if ok else 1)
