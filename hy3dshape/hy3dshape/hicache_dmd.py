"""Compatibility facade for central cached-fit DMD/Prony forecasting."""

try:
    from hicache_pp.dmd import (
        dmd_eval,
        dmd_fit,
        dmd_forecast,
        dmd_forecast_state,
        dmd_update_snapshots,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "hunyuan2.1-plus-plus requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


__all__ = [
    "dmd_eval", "dmd_fit", "dmd_forecast", "dmd_forecast_state",
    "dmd_update_snapshots",
]


if __name__ == "__main__":
    import torch

    snapshots = [torch.tensor([0.9 ** step, 0.7 ** step]) for step in range(4)]
    state = {
        "step": 5,
        "history": 5,
        "dmd_snapshots": [(step, value) for step, value in enumerate(snapshots)],
        "derivatives": {0: snapshots[-1]},
        "activated_steps": [3],
    }
    assert torch.isfinite(dmd_forecast_state(state)).all()
    state["step"] = 6
    assert torch.isfinite(dmd_forecast_state(state)).all()
    assert state.get("_dmd_fit_key") == (3, 4, 1)
    print("hunyuan2.1-plus-plus central DMD smoke passed")
