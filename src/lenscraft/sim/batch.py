"""``simulate_lens_batch``: execute an approved LensCard and write a lensjsonl dataset.

Two layers so the same code runs locally and sharded across Modal containers:

- :func:`simulate_range` renders images ``[start, stop)`` of a card into a run directory and
  returns their records without touching the records file. Every image gets its own RNG seeded
  from ``card.seed`` and its *global* index, so shards rendered on different machines are
  bit-for-bit identical to one local run.
- :func:`simulate_lens_batch` renders the whole card (or a slice) and writes ``card.json`` and
  ``records.lensjsonl``. This is what the agent's tool calls.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from lenscraft.schema.lens_card import LensCard, SimBatchResult
from lenscraft.schema.lensjsonl import LensRecord, write_records
from lenscraft.sim.stats import compute_moment_stats, estimate_snr

RECORDS_FILENAME = "records.lensjsonl"
CARD_FILENAME = "card.json"


def _get_backend(name: str):
    if name == "lenstronomy":
        from lenscraft.sim import lenstronomy_backend

        return lenstronomy_backend.simulate_one
    if name == "pyautolens":
        raise NotImplementedError("pyautolens backend is a Phase 5 stretch goal (DESIGN.md Section 11)")
    raise ValueError(f"unknown backend {name!r}")


def image_rng(card: LensCard, index: int) -> np.random.Generator:
    """RNG for image ``index`` of ``card``; equals ``SeedSequence(seed).spawn(n)[index]`` for any n."""
    return np.random.default_rng(np.random.SeedSequence(card.seed, spawn_key=(index,)))


def simulate_range(
    card: LensCard,
    run_dir: str | Path,
    run_id: str,
    start: int = 0,
    stop: int | None = None,
    *,
    progress: bool = False,
) -> list[LensRecord]:
    """Render images ``[start, stop)`` into ``<run_dir>/images`` and return their records."""
    stop = card.n_images if stop is None else min(stop, card.n_images)
    if not 0 <= start <= stop:
        raise ValueError(f"invalid range [{start}, {stop}) for n_images={card.n_images}")
    run_dir = Path(run_dir)
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    simulate_one = _get_backend(card.backend)

    records: list[LensRecord] = []
    for i in range(start, stop):
        rendered = simulate_one(card, image_rng(card, i))
        image_name = f"{i:05d}.npy"
        np.save(image_dir / image_name, rendered.image)

        stats = compute_moment_stats(rendered.model, rendered.smooth_model, rendered.pixel_scale)
        snr = estimate_snr(rendered.model, rendered.noise_sigma)
        sub_mass = rendered.realised.get("substructure_mass", 0.0)
        records.append(
            LensRecord(
                image_id=f"{run_id}_{i:05d}",
                lens_card=card,
                substructure_type=card.substructure,
                mass_fraction=float(min(1.0, sub_mass / card.halo_mass)),
                snr=snr,
                image_path=f"images/{image_name}",
                moment_stats=stats,
                extras=rendered.realised,
            )
        )
        if progress:
            print(f"[{run_id}] {i + 1}/{card.n_images} snr={snr:.1f}", flush=True)
    return records


def write_run(run_dir: str | Path, card: LensCard, records: Iterable[LensRecord]) -> Path:
    """Write ``card.json`` and ``records.lensjsonl`` for a run; returns the records path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CARD_FILENAME).write_text(card.model_dump_json(indent=2), encoding="utf-8")
    records_path = run_dir / RECORDS_FILENAME
    write_records(records_path, records)
    return records_path


def simulate_lens_batch(
    card: LensCard,
    out_dir: str | Path,
    run_id: str,
    *,
    progress: bool = False,
) -> SimBatchResult:
    """Generate ``card.n_images`` images and write ``<out_dir>/<run_id>/records.lensjsonl``.

    Layout::

        <out_dir>/<run_id>/
            card.json               the LensCard that produced this run
            records.lensjsonl       one LensRecord per image
            images/00000.npy ...    float32 observed images, shape (image_size, image_size)

    Failures are returned as ``status="failed"`` with the error in ``message`` rather than
    raised, so an agent sees a structured result either way. Whatever was rendered before the
    failure is still written.
    """
    t0 = time.perf_counter()
    run_dir = Path(out_dir) / run_id
    records: list[LensRecord] = []
    try:
        records = simulate_range(card, run_dir, run_id, progress=progress)
    except Exception as exc:  # noqa: BLE001 - surface to the agent as a structured failure
        records_path = write_run(run_dir, card, records)
        return SimBatchResult(
            status="failed",
            run_id=run_id,
            lensjsonl_path=str(records_path),
            image_dir=str(run_dir / "images"),
            n_images=len(records),
            elapsed_seconds=time.perf_counter() - t0,
            message=f"{type(exc).__name__}: {exc}",
        )

    records_path = write_run(run_dir, card, records)
    return SimBatchResult(
        status="ok",
        run_id=run_id,
        lensjsonl_path=str(records_path),
        image_dir=str(run_dir / "images"),
        n_images=len(records),
        elapsed_seconds=time.perf_counter() - t0,
    )


def load_run_card(run_dir: str | Path) -> LensCard:
    return LensCard.model_validate(json.loads((Path(run_dir) / CARD_FILENAME).read_text(encoding="utf-8")))
