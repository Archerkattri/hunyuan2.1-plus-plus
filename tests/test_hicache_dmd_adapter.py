"""CPU tests for the Hunyuan3D-2.1 central DMD adapter."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hicache = _load("hunyuan21_hicache_adapter", "hy3dshape/hy3dshape/hicache.py")
dmd = _load("hunyuan21_hicache_dmd_adapter", "hy3dshape/hy3dshape/hicache_dmd.py")


def _trajectory():
    return [torch.tensor([0.9 ** step, 0.7 ** step]) for step in range(6)]


def _state(values):
    state = hicache.hicache_init(num_steps=20, interval=1, first_enhance=0,
                                 backend="dmd", history=5)
    for step, value in enumerate(values):
        state["step"] = step
        state["activated_steps"].append(step)
        hicache.hicache_update_derivatives(state, value)
        dmd.dmd_update_snapshots(state, value, history=5)
    return state


def test_facades_match_central_and_fit_is_reused():
    import hicache_pp.dmd as central_dmd
    import hicache_pp.hermite as central_hermite

    assert dmd.dmd_forecast_state is central_dmd.dmd_forecast_state
    assert hicache.hicache_forecast is central_hermite.hicache_forecast

    calls = 0
    original = central_dmd.dmd_fit
    state = _state(_trajectory()[:5])

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    central_dmd.dmd_fit = counted
    try:
        for step in (5, 6, 7):
            state["step"] = step
            assert torch.isfinite(dmd.dmd_forecast_state(state)).all()
    finally:
        central_dmd.dmd_fit = original
    assert calls == 1
    assert state["_dmd_fit_key"] == (4, 5, 1)
    assert state["telemetry"]["method_counts"]["dmd"] == 3


def test_snapshot_ownership_and_reset_between_two_seed_runs():
    state_a = _state(_trajectory()[:4])
    run_a = state_a["run_id"]
    stored = state_a["dmd_snapshots"][-1][1]
    source = stored.clone().requires_grad_(True)
    state_a["activated_steps"].append(4)
    dmd.dmd_update_snapshots(state_a, source)
    with torch.no_grad():
        source.add_(10.0)
    assert not state_a["dmd_snapshots"][-1][1].requires_grad
    assert torch.equal(state_a["dmd_snapshots"][-1][1], stored)

    state_b = hicache.hicache_init(num_steps=20, interval=1, backend="dmd")
    assert state_b["run_id"] != run_a
    hicache.hicache_reset(state_a)
    assert state_a["run_id"] != run_a
    assert state_a["dmd_snapshots"] == []
    assert "_dmd_fit_key" not in state_a
    assert hicache.hicache_telemetry(state_a)["decisions"] == {"full": 0, "forecast": 0}


def test_short_or_nonuniform_history_falls_back_to_hermite():
    traj = _trajectory()
    state = hicache.hicache_init(num_steps=10, interval=3, first_enhance=0,
                                 backend="dmd", max_order=1, sigma=0.5)
    state["step"] = 4
    state["activated_steps"] = [3]
    hicache.hicache_update_derivatives(state, traj[3])
    state["dmd_snapshots"] = [(1, traj[1]), (3, traj[3])]
    output = dmd.dmd_forecast_state(state)
    assert torch.equal(output, traj[3])
    assert state["telemetry"]["fallbacks"]["dmd_insufficient_uniform_history"] == 1


def test_native_cfg_combined_formula_and_adaptive_index_contract():
    cond, uncond, scale = torch.tensor([3.0]), torch.tensor([1.0]), 5.0
    # Pipeline computes uncond + scale*(cond-uncond), then records adaptive-CFG
    # anchors against the real diffusion index i, not the cache counter.
    assert torch.equal(uncond + scale * (cond - uncond), torch.tensor([11.0]))
    state = {"step": 0, "last_gamma": None, "anchors": [], "n_full": 0, "n_skip": 0}
    for i in (0, 3, 6):
        state["step"] = i
        state["anchors"].append((i, torch.tensor([float(i)])))
    assert [step for step, _ in state["anchors"]] == [0, 3, 6]
