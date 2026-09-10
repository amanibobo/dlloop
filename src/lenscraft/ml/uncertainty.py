"""sample_uncertainty: where in LensCard space is the current model weakest? (DESIGN.md Section 8)

Pure numpy over the per-image scores written by ``evaluate_model`` joined with the lensjsonl
records, so it runs anywhere (laptop or agent process) without torch. The result is an
``UncertaintyReport`` whose ``suggested_card`` targets the weakest region; the agent submits that
through the usual approval gate, which is what closes the simulate -> train -> resimulate loop.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from lenscraft.schema.lens_card import LensCard
from lenscraft.schema.lensjsonl import LensRecord
from lenscraft.schema.ml import CLASS_INDEX, CLASS_NAMES, EvalSpec
from lenscraft.schema.uncertainty import UncertaintyCell, UncertaintyReport

AXES_BY_CLASS: dict[str, list[str]] = {
    "none": ["snr"],
    "subhalo": ["mass_fraction", "n_subhalos", "snr"],
    "vortex": ["mass_fraction", "axion_mass", "snr"],
}
LOG_AXES = {"axion_mass"}
NONE_FALSE_ALARM_QUANTILE = 0.95


# --------------------------------------------------------------------------------------------
# Per-image uncertainty
# --------------------------------------------------------------------------------------------


def per_image_uncertainty(kind: str, images: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (uncertainty in [0,1], success in {0,1}) per image, in the order given.

    classifier: uncertainty = 1 - P(true class); success = argmax == true class.
    anomaly:    scores are reconstruction errors. threshold = 95th percentile of 'none' scores.
                substructure images: uncertainty = fraction of 'none' images scoring at least as
                high (how ordinary this anomaly looks); success = score > threshold (detected).
                'none' images: uncertainty = fraction of 'none' images scoring lower (how
                outlying it is); success = score <= threshold (no false alarm).
    """
    if kind == "classifier":
        probs = np.array([im["probs"] for im in images], dtype=float)
        labels = np.array([CLASS_INDEX[im["label"]] for im in images])
        u = 1.0 - probs[np.arange(len(images)), labels]
        ok = (probs.argmax(1) == labels).astype(float)
        return np.clip(u, 0, 1), ok

    scores = np.array([im["score"] for im in images], dtype=float)
    labels = np.array([im["label"] for im in images])
    none_scores = np.sort(scores[labels == "none"])
    if len(none_scores) == 0:
        raise ValueError("anomaly uncertainty needs some substructure='none' images in the evaluated runs")
    thr = float(np.quantile(none_scores, NONE_FALSE_ALARM_QUANTILE))
    frac_none_ge = 1.0 - np.searchsorted(none_scores, scores, side="left") / len(none_scores)
    u = np.where(labels == "none", 1.0 - frac_none_ge, frac_none_ge)
    ok = np.where(labels == "none", scores <= thr, scores > thr).astype(float)
    return np.clip(u, 0, 1), ok


# --------------------------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------------------------


def _axis_value(rec: LensRecord, axis: str) -> float | None:
    if axis == "mass_fraction":
        return rec.mass_fraction
    if axis == "snr":
        return rec.snr
    v = rec.extras.get(axis)
    return None if v is None else float(v)


def _bin_edges(values: np.ndarray, n_bins: int, log: bool) -> np.ndarray:
    v = np.round(np.log10(values), 6) if log else np.round(values, 9)  # collapse float noise (0.03 vs 0.030000000004)
    uniq = np.unique(v)
    if len(uniq) < 2:
        return np.array([uniq[0], uniq[0]]) if len(uniq) else np.array([0.0, 0.0])
    edges = np.unique(np.quantile(v, np.linspace(0, 1, min(n_bins, len(uniq)) + 1)))
    return edges


def build_cells(records: list[LensRecord], u: np.ndarray, ok: np.ndarray, n_bins: int) -> list[UncertaintyCell]:
    cells: list[UncertaintyCell] = []
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_class[r.substructure_type].append(i)
    for cls, idx in by_class.items():
        idx_arr = np.array(idx)
        for axis in AXES_BY_CLASS.get(cls, ["snr"]):
            vals = np.array([_axis_value(records[i], axis) for i in idx_arr], dtype=object)
            keep = np.array([v is not None for v in vals])
            if not keep.any():
                continue
            ii = idx_arr[keep]
            v = np.array([float(x) for x in vals[keep]])
            log = axis in LOG_AXES
            edges = _bin_edges(v, n_bins, log)
            if len(edges) < 3 and len(AXES_BY_CLASS.get(cls, ["snr"])) > 1:
                continue  # a single-valued axis says nothing about *where* the model is weak
            tv = np.round(np.log10(v), 6) if log else np.round(v, 9)
            for b in range(max(1, len(edges) - 1)):
                lo, hi = edges[b], edges[b + 1] if len(edges) > 1 else edges[b]
                m = (tv >= lo) & ((tv <= hi) if b == len(edges) - 2 or len(edges) < 2 else (tv < hi))
                if not m.any():
                    continue
                sel = ii[m]
                cells.append(
                    UncertaintyCell(
                        substructure=cls, axis=axis,
                        low=float(10**lo if log else lo), high=float(10**hi if log else hi),
                        n=int(m.sum()), mean_uncertainty=float(u[sel].mean()), score=float(ok[sel].mean()),
                    )
                )
    return cells


# --------------------------------------------------------------------------------------------
# Proposal
# --------------------------------------------------------------------------------------------


def propose_card(cell: UncertaintyCell, records: list[LensRecord], n_images: int, seed: int | None) -> tuple[LensCard, str]:
    """A LensCard that adds images in the weakest cell, derived from the card of the run that
    contributed most images of that class."""
    counts: dict[str, int] = defaultdict(int)
    cards: dict[str, LensCard] = {}
    for r in records:
        if r.substructure_type == cell.substructure:
            run = r.image_id.rsplit("_", 1)[0]
            counts[run] += 1
            cards[run] = r.lens_card
    base = cards[max(counts, key=counts.get)]
    if seed is None and base.seed is not None:
        seed = base.seed + 1000  # reproducible and distinct from the run it was derived from
    fields: dict[str, Any] = {"n_images": n_images, "seed": seed}
    note = ""
    if cell.axis == "mass_fraction" and cell.substructure != "none":
        mid = 0.5 * (cell.low + cell.high)
        fields["substructure_mass_fraction"] = float(min(1.0, max(1e-4, mid)))
        note = f"substructure_mass_fraction set to the cell midpoint {fields['substructure_mass_fraction']:.4g}"
    elif cell.axis == "axion_mass":
        mid = math.sqrt(cell.low * cell.high)
        fields["axion_mass"] = float(mid)
        note = f"axion_mass fixed at the cell's geometric midpoint {mid:.3g} eV (vortex length {0.06 / (2 * mid) * 1e-22:.3g}\")"
    elif cell.axis == "n_subhalos":
        fields["n_subhalos"] = max(1, int(round(0.5 * (cell.low + cell.high))))
        note = f"n_subhalos set to {fields['n_subhalos']}"
    elif cell.axis == "snr":
        note = (
            f"SNR is not a LensCard field (it follows from source_magnitude / exposure); the proposal keeps the base card "
            f"and adds more {cell.substructure} images so the SNR range [{cell.low:.0f}, {cell.high:.0f}] is better covered"
        )
    card = LensCard.model_validate({**base.model_dump(), **fields})
    return card, note


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def analyze(
    model_id: str,
    kind: str,
    run_ids: list[str],
    scores: dict[str, Any],
    records: list[LensRecord],
    *,
    n_bins: int = 4,
    min_cell_n: int = 10,
    n_images_proposed: int = 2000,
    seed: int | None = None,
    top_k: int = 12,
) -> UncertaintyReport:
    by_id = {r.image_id: r for r in records}
    images = [im for im in scores["images"] if im["image_id"] in by_id]
    if not images:
        return UncertaintyReport(status="failed", model_id=model_id, kind=kind, run_ids=run_ids, message="no scored images match the run records")
    recs = [by_id[im["image_id"]] for im in images]
    u, ok = per_image_uncertainty(kind, images)

    per_class = {}
    for cls in CLASS_NAMES:
        m = np.array([r.substructure_type == cls for r in recs])
        if m.any():
            per_class[cls] = float(u[m].mean())

    cells = build_cells(recs, u, ok, n_bins)
    cells.sort(key=lambda c: (-c.mean_uncertainty, -c.n))
    eligible = [c for c in cells if c.n >= min_cell_n] or cells
    weakest = eligible[0] if eligible else None

    suggested, note = (None, "")
    if weakest is not None:
        suggested, note = propose_card(weakest, recs, n_images_proposed, seed)
    rationale = ""
    if weakest is not None:
        what = "accuracy" if kind == "classifier" else "detection rate"
        rationale = (
            f"Weakest region: {weakest.substructure} with {weakest.axis} in [{weakest.low:.4g}, {weakest.high:.4g}] "
            f"(n={weakest.n}, mean uncertainty {weakest.mean_uncertainty:.3f}, {what} {weakest.score:.3f}). "
            f"Proposed {n_images_proposed} more {weakest.substructure} images there; {note}."
        )
    return UncertaintyReport(
        model_id=model_id, kind=kind, run_ids=run_ids, n_images=len(images),
        overall_uncertainty=float(u.mean()), per_class_uncertainty=per_class,
        cells=eligible[:top_k], weakest=weakest, suggested_card=suggested, rationale=rationale,
    )


def sample_uncertainty(
    backend,
    model_id: str,
    run_ids: list[str],
    *,
    n_bins: int = 4,
    min_cell_n: int = 10,
    n_images_proposed: int = 2000,
    seed: int | None = None,
) -> UncertaintyReport:
    """Read (or produce) the model's per-image scores on ``run_ids`` and analyse them."""
    scores = backend.read_eval_scores(model_id, run_ids)
    if scores is None:
        ev = backend.evaluate(EvalSpec(model_id=model_id, run_ids=run_ids))
        if ev.status != "ok":
            return UncertaintyReport(status="failed", model_id=model_id, run_ids=run_ids, message=f"evaluation failed: {ev.message}")
        scores = backend.read_eval_scores(model_id, run_ids)
        if scores is None:
            return UncertaintyReport(status="failed", model_id=model_id, run_ids=run_ids, message="evaluation produced no scores file")
    records: list[LensRecord] = []
    for run_id in run_ids:
        records.extend(backend.read_records(run_id))
    return analyze(model_id, scores["kind"], run_ids, scores, records, n_bins=n_bins, min_cell_n=min_cell_n, n_images_proposed=n_images_proposed, seed=seed)
