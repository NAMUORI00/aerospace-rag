from __future__ import annotations

from pathlib import Path

from .fusion import resolve_enterprise_weights


def resolve_channel_weights(
    query: str,
    *,
    profile_path: str | Path,
    profile_meta_path: str | Path | None = None,
    mode: str = "hybrid",
    min_weight: float = 0.10,
    max_weight: float = 0.80,
) -> tuple[dict[str, float], dict[str, object]]:
    weights, diagnostics = resolve_enterprise_weights(
        query,
        profile_path=Path(profile_path),
        profile_meta_path=Path(profile_meta_path) if profile_meta_path else None,
        mode=mode,
    )
    return weights, diagnostics
