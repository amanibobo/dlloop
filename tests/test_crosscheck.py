"""Cross-backend validation. The real comparison needs the optional 'autolens' package (run
`uv run --python .venv-autolens/bin/python -m pytest tests/test_crosscheck.py`); without it the
tests only check the structured-failure path."""

import numpy as np
import pytest

from lenscraft.schema import LensCard
from lenscraft.sim.crosscheck import cross_backend_validate

lenstronomy = pytest.importorskip("lenstronomy")
pytestmark = pytest.mark.integration

try:
    import autolens  # noqa: F401

    HAVE_AL = True
except ImportError:
    HAVE_AL = False


def test_euclid_instrument_is_rejected():
    r = cross_backend_validate(LensCard(n_images=1, substructure="none"), 1)
    assert r.status == "failed" and "custom" in r.message


@pytest.mark.skipif(HAVE_AL, reason="autolens installed: the real path is tested below")
def test_missing_autolens_is_structured(tmp_path):
    from lenscraft.sim.batch import simulate_lens_batch

    r = cross_backend_validate(LensCard(n_images=1, substructure="none", instrument="custom", image_size=64), 1)
    assert r.status == "failed" and "autolens" in r.message
    res = simulate_lens_batch(LensCard(n_images=1, substructure="none", instrument="custom", image_size=64, backend="pyautolens"), tmp_path, "p")
    assert res.status == "failed" and "autolens" in res.message


@pytest.mark.skipif(not HAVE_AL, reason="needs autolens")
@pytest.mark.parametrize("substructure", ["none", "subhalo", "vortex"])
def test_engines_agree(substructure):
    card = LensCard(n_images=2, substructure=substructure, instrument="custom", image_size=64, pixel_scale=0.1, seed=5)
    r = cross_backend_validate(card, 2)
    assert r.status == "ok", r.message
    assert r.n_images == 2 and r.agree, r.model_dump()
    assert r.max_shape_rel_rms < 0.05 and abs(r.mean_flux_ratio - 1) < 0.1
    assert all(im.centroid_shift_pixels <= 0.5 for im in r.images)


@pytest.mark.skipif(not HAVE_AL, reason="needs autolens")
def test_pyautolens_backend_renders_dataset(tmp_path):
    from lenscraft.schema import read_records
    from lenscraft.sim.batch import simulate_lens_batch

    card = LensCard(n_images=2, substructure="vortex", instrument="custom", image_size=64, pixel_scale=0.1, seed=1, backend="pyautolens")
    res = simulate_lens_batch(card, tmp_path, "pal")
    assert res.status == "ok", res.message
    recs = read_records(res.lensjsonl_path)
    img = np.load(tmp_path / "pal" / recs[0].image_path)
    assert img.shape == (64, 64) and np.isfinite(img).all() and recs[0].snr > 0
    assert recs[0].moment_stats.residual_rms > 0  # substructure perturbs the arcs in this engine too
