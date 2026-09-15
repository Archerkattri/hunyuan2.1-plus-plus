"""Compatibility facade for the shared scalar HiCache Hermite core.

The Hunyuan3D-2.1 pipeline owns CFG/adaptive-guidance indexing and calls these
helpers at its native post-CFG update site. The model-agnostic Hermite state,
corrected-sign arithmetic, reset, and telemetry come from ``hicache-pp``.
"""

try:
    from hicache_pp.hermite import (
        hicache_decide,
        hicache_forecast,
        hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_derivatives,
        physicists_hermite,
        scaled_hermite,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "hunyuan2.1-plus-plus requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


__all__ = [
    "hicache_decide", "hicache_forecast", "hicache_init", "hicache_reset",
    "hicache_telemetry", "hicache_update_derivatives", "physicists_hermite",
    "scaled_hermite",
]
