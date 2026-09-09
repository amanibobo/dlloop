"""Modal app: sharded simulation onto a persistent Volume, off the laptop entirely.

Deploy once (after ``modal setup``)::

    modal deploy -m lenscraft.compute.modal_app

Smoke test from the laptop (runs remotely, prints the SimBatchResult)::

    modal run -m lenscraft.compute.modal_app --substructure vortex --n 50 --run-id smoke

Layout on the ``lenscraft-data`` Volume mirrors the local layout: ``/data/<run_id>/card.json``,
``records.lensjsonl``, ``images/*.npy``. A batch is split into shards of ``SHARD_SIZE`` images
that render in parallel containers; per-image seeding makes the result identical to a local run.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

APP_NAME = "lenscraft"
VOLUME_NAME = "lenscraft-data"
DATA_DIR = "/data"
SHARD_SIZE = 500

app = modal.App(APP_NAME)

sim_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pydantic>=2.7", "numpy>=1.26", "scipy>=1.12", "astropy>=6.0", "lenstronomy>=1.12")
    .add_local_python_source("lenscraft")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=sim_image, volumes={DATA_DIR: volume}, cpu=2.0, timeout=2 * 3600)
def simulate_shard(card_json: str, run_id: str, start: int, stop: int) -> list[str]:
    """Render images [start, stop) of the card into the Volume; return their records as JSON lines."""
    from lenscraft.schema import LensCard
    from lenscraft.sim.batch import simulate_range

    card = LensCard.model_validate_json(card_json)
    records = simulate_range(card, Path(DATA_DIR) / run_id, run_id, start, stop)
    volume.commit()
    return [r.model_dump_json() for r in records]


@app.function(image=sim_image, volumes={DATA_DIR: volume}, timeout=6 * 3600)
def simulate_batch(card_json: str, run_id: str) -> dict:
    """Fan a LensCard out over shards, merge the records, and return a SimBatchResult dict."""
    from lenscraft.schema import LensCard, LensRecord, SimBatchResult
    from lenscraft.sim.batch import write_run

    t0 = time.perf_counter()
    card = LensCard.model_validate_json(card_json)
    run_dir = Path(DATA_DIR) / run_id
    image_dir = run_dir / "images"
    shards = [(card_json, run_id, s, min(s + SHARD_SIZE, card.n_images)) for s in range(0, card.n_images, SHARD_SIZE)]

    lines: list[str] = []
    status, message = "ok", ""
    try:
        for out in simulate_shard.starmap(shards):
            lines.extend(out)
    except Exception as exc:  # noqa: BLE001 - report as a structured failure
        status, message = "failed", f"{type(exc).__name__}: {exc}"

    records = sorted((LensRecord.model_validate_json(line) for line in lines), key=lambda r: r.image_id)
    records_path = write_run(run_dir, card, records)
    volume.commit()
    return SimBatchResult(
        status=status,
        run_id=run_id,
        lensjsonl_path=str(records_path),
        image_dir=str(image_dir),
        n_images=len(records),
        elapsed_seconds=time.perf_counter() - t0,
        message=message,
    ).model_dump()


@app.function(image=sim_image, volumes={DATA_DIR: volume})
def read_run(run_id: str) -> str:
    """Return the records.lensjsonl text for a run (small: ~0.5 KB per image)."""
    volume.reload()
    path = Path(DATA_DIR) / run_id / "records.lensjsonl"
    if not path.exists():
        raise FileNotFoundError(f"no run {run_id!r} on volume {VOLUME_NAME}")
    return path.read_text(encoding="utf-8")


@app.function(image=sim_image, volumes={DATA_DIR: volume})
def list_runs() -> list[str]:
    volume.reload()
    root = Path(DATA_DIR)
    return sorted(p.name for p in root.iterdir() if (p / "records.lensjsonl").exists())


@app.local_entrypoint()
def main(substructure: str = "none", n: int = 10, seed: int = 0, run_id: str = "smoke") -> None:
    from lenscraft.schema import LensCard

    card = LensCard(n_images=n, substructure=substructure, seed=seed)  # type: ignore[arg-type]
    print(simulate_batch.remote(card.model_dump_json(), run_id))
