"""lensjsonl: one JSON record per simulated image (DESIGN.md Section 4.2).

This is the LLM-legible intermediate format the agent reasons over instead of raw pixels,
mirroring HEPTAPOD's ``evtjsonl``. Image arrays live in ``.npy`` files next to it.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lenscraft.schema.lens_card import LensCard, SubstructureType


class MomentStats(BaseModel):
    """Cheap image statistics computed at generation time.

    - ``residual_rms``: rms of (full noiseless model - smooth-lens-only noiseless model), divided
      by the smooth model's peak. Measures how strongly the substructure perturbs the arcs; it is
      exactly 0 for the 'none' class.
    - ``second_moment``: sqrt of the trace of the flux-weighted second-moment tensor, in arcsec.
      A characteristic size of the lensed light.
    - ``arc_ellipticity``: |e| from the same tensor, e1 = (Qxx - Qyy)/(Qxx + Qyy), e2 = 2Qxy/(Qxx + Qyy).
    """

    model_config = ConfigDict(extra="forbid")

    residual_rms: float = Field(ge=0)
    second_moment: float = Field(ge=0)
    arc_ellipticity: float = Field(ge=0, le=1)


class LensRecord(BaseModel):
    """One line of a ``.lensjsonl`` file."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    lens_card: LensCard
    substructure_type: SubstructureType
    mass_fraction: float = Field(ge=0, le=1, description="Realised substructure mass / halo mass.")
    snr: float = Field(ge=0, description="Measured signal-to-noise of the lensed light.")
    image_path: str = Field(description="Path to the .npy image, relative to the lensjsonl file's directory.")
    moment_stats: MomentStats
    extras: dict[str, float] = Field(
        default_factory=dict,
        description="Per-image realised parameters (theta_E, exposure_time, axion_mass, n_subhalos, ...).",
    )


def write_records(path: str | Path, records: Iterable[LensRecord]) -> int:
    """Write records as JSON lines. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def iter_records(path: str | Path) -> Iterator[LensRecord]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield LensRecord.model_validate_json(line)
            except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
                raise ValueError(f"{path}:{line_no}: invalid lensjsonl record: {exc}") from exc


def read_records(path: str | Path) -> list[LensRecord]:
    return list(iter_records(path))


def summarize_records(records: Iterable[LensRecord]) -> dict[str, Any]:
    """Dataset-composition summary suitable for dropping into an agent's context."""
    records = list(records)
    if not records:
        return {"n_images": 0, "by_class": {}, "snr": None, "mass_fraction": None}

    def _stats(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    by_class = Counter(r.substructure_type for r in records)
    per_class: dict[str, dict[str, Any]] = {}
    for cls in sorted(by_class):
        rs = [r for r in records if r.substructure_type == cls]
        per_class[cls] = {
            "n": len(rs),
            "snr": _stats([r.snr for r in rs]),
            "mass_fraction": _stats([r.mass_fraction for r in rs]),
            "residual_rms": _stats([r.moment_stats.residual_rms for r in rs]),
        }
    return {
        "n_images": len(records),
        "by_class": per_class,
        "snr": _stats([r.snr for r in records]),
        "mass_fraction": _stats([r.mass_fraction for r in records]),
        "runs": sorted({r.image_id.rsplit("_", 1)[0] for r in records}),
    }


def dumps_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
