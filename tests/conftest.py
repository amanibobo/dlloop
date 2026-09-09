"""Shared fixtures."""

import numpy as np
import pytest

from lenscraft.schema import LensCard, LensRecord, MomentStats
from lenscraft.sim.batch import write_run


def make_synthetic_run(root, run_id: str, substructure: str, n: int, size: int = 32, seed: int = 0) -> list[LensRecord]:
    """Write a fake run whose images carry a class-dependent pattern so a model can learn something.

    none: a ring; subhalo: ring + bright blob; vortex: ring + a line. No Lenstronomy needed.
    """
    rng = np.random.default_rng(seed)
    run_dir = root / run_id
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    card = LensCard(n_images=n, substructure=substructure, image_size=size, seed=seed)
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.hypot(x - c, y - c)
    records = []
    for i in range(n):
        img = np.exp(-0.5 * ((r - size / 4) / 2.0) ** 2)
        if substructure == "subhalo":
            bx, by = rng.integers(4, size - 4, size=2)
            img += 1.5 * np.exp(-0.5 * (np.hypot(x - bx, y - by) / 1.5) ** 2)
        elif substructure == "vortex":
            ang = rng.uniform(0, np.pi)
            d = np.abs((x - c) * np.sin(ang) - (y - c) * np.cos(ang))
            img += 1.0 * np.exp(-0.5 * (d / 1.0) ** 2) * (r < size / 3)
        img = img + rng.normal(0, 0.05, img.shape)
        np.save(run_dir / "images" / f"{i:05d}.npy", img.astype(np.float32))
        records.append(
            LensRecord(
                image_id=f"{run_id}_{i:05d}", lens_card=card, substructure_type=substructure,
                mass_fraction=0.0 if substructure == "none" else 0.01, snr=100.0, image_path=f"images/{i:05d}.npy",
                moment_stats=MomentStats(residual_rms=0.0, second_moment=0.5, arc_ellipticity=0.1),
            )
        )
    write_run(run_dir, card, records)
    return records


@pytest.fixture
def synthetic_dataset(tmp_path):
    """tmp_path with runs none_a, sub_a, vor_a (12 images each) and held-out none_b, vor_b (6 each)."""
    for run_id, cls, n in [("none_a", "none", 12), ("sub_a", "subhalo", 12), ("vor_a", "vortex", 12), ("none_b", "none", 6), ("vor_b", "vortex", 6)]:
        make_synthetic_run(tmp_path, run_id, cls, n, seed=hash(run_id) % 1000)
    return tmp_path
