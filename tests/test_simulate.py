"""End-to-end checks against the real Lenstronomy engine (small images, few samples)."""

import numpy as np
import pytest

lenstronomy = pytest.importorskip("lenstronomy")

from lenscraft.schema import LensCard, read_records  # noqa: E402
from lenscraft.sim import simulate_lens_batch  # noqa: E402
from lenscraft.sim.batch import load_run_card  # noqa: E402
from lenscraft.sim.cosmo import point_mass_einstein_radius  # noqa: E402

pytestmark = pytest.mark.integration


def test_einstein_radius_is_physical():
    # 1e12 Msun at z=0.5 lensing z=1 -> ~1.66" with angular-diameter distances
    assert point_mass_einstein_radius(1e12, 0.5, 1.0) == pytest.approx(1.66, abs=0.05)
    arr = point_mass_einstein_radius(np.array([1e8, 1e10]), 0.5, 1.0)
    assert arr.shape == (2,) and arr[1] == pytest.approx(arr[0] * 10, rel=1e-9)


@pytest.mark.parametrize("substructure", ["none", "subhalo", "vortex"])
@pytest.mark.parametrize("instrument", ["euclid", "custom"])
def test_batch_writes_valid_dataset(tmp_path, substructure, instrument):
    # Euclid is 0.101"/px, custom defaults to 0.05"/px: both give a 3.2" field, enough for theta_E ~1.66".
    size = 32 if instrument == "euclid" else 64
    card = LensCard(n_images=2, substructure=substructure, instrument=instrument, image_size=size, seed=7)
    result = simulate_lens_batch(card, tmp_path, run_id="t")
    assert result.status == "ok", result.message
    assert result.n_images == 2

    records = read_records(result.lensjsonl_path)
    assert len(records) == 2
    assert load_run_card(tmp_path / "t") == card
    for rec in records:
        assert rec.substructure_type == substructure
        img = np.load(tmp_path / "t" / rec.image_path)
        assert img.shape == (size, size) and img.dtype == np.float32
        assert np.isfinite(img).all()
        assert rec.snr > 0
        assert "theta_E" in rec.extras and "exposure_time" in rec.extras
        assert rec.extras["field_of_view"] == pytest.approx(3.2, abs=0.05)
        if substructure == "none":
            assert rec.mass_fraction == 0.0 and rec.moment_stats.residual_rms == 0.0
        else:
            assert rec.mass_fraction == pytest.approx(0.01, rel=1e-6)
            assert rec.moment_stats.residual_rms > 0.0
        if substructure == "vortex":
            assert 1e-24 <= rec.extras["axion_mass"] <= 1e-22
        if substructure == "subhalo":
            assert rec.extras["n_subhalos"] >= 1


def test_seed_reproducible(tmp_path):
    card = LensCard(n_images=2, substructure="subhalo", image_size=32, seed=123)
    a = simulate_lens_batch(card, tmp_path / "a", run_id="r")
    b = simulate_lens_batch(card, tmp_path / "b", run_id="r")
    ra, rb = read_records(a.lensjsonl_path), read_records(b.lensjsonl_path)
    assert [r.extras for r in ra] == [r.extras for r in rb]
    for x, y in zip(ra, rb):
        np.testing.assert_array_equal(np.load(tmp_path / "a" / "r" / x.image_path), np.load(tmp_path / "b" / "r" / y.image_path))


def test_seed_per_image_independent_of_batch_size(tmp_path):
    small = simulate_lens_batch(LensCard(n_images=1, substructure="none", image_size=32, seed=5), tmp_path / "s", run_id="r")
    big = simulate_lens_batch(LensCard(n_images=3, substructure="none", image_size=32, seed=5), tmp_path / "b", run_id="r")
    x = np.load(tmp_path / "s" / "r" / read_records(small.lensjsonl_path)[0].image_path)
    y = np.load(tmp_path / "b" / "r" / read_records(big.lensjsonl_path)[0].image_path)
    np.testing.assert_array_equal(x, y)


def test_sharded_ranges_match_single_run(tmp_path):
    from lenscraft.sim.batch import simulate_range

    card = LensCard(n_images=3, substructure="vortex", image_size=32, seed=11)
    full = simulate_lens_batch(card, tmp_path / "full", run_id="r")
    a = simulate_range(card, tmp_path / "shard" / "r", "r", 0, 2)
    b = simulate_range(card, tmp_path / "shard" / "r", "r", 2, 3)
    assert [r.extras for r in a + b] == [r.extras for r in read_records(full.lensjsonl_path)]
    for i in range(3):
        np.testing.assert_array_equal(
            np.load(tmp_path / "full" / "r" / "images" / f"{i:05d}.npy"),
            np.load(tmp_path / "shard" / "r" / "images" / f"{i:05d}.npy"),
        )


def test_fixed_axion_mass_sets_vortex_length(tmp_path):
    card = LensCard(n_images=1, substructure="vortex", axion_mass=1e-23, image_size=32, seed=1)
    result = simulate_lens_batch(card, tmp_path, run_id="v")
    rec = read_records(result.lensjsonl_path)[0]
    assert rec.extras["axion_mass"] == 1e-23
    assert rec.extras["vortex_half_length"] == pytest.approx(0.3)


def test_undersized_field_of_view_is_a_structured_failure(tmp_path):
    # 32 px x 0.05"/px = 1.6" cannot hold a 3.3" Einstein ring; must not silently produce noise
    card = LensCard(n_images=1, substructure="none", instrument="custom", image_size=32, seed=1)
    result = simulate_lens_batch(card, tmp_path, run_id="f")
    assert result.status == "failed"
    assert result.n_images == 0
    assert "field of view" in result.message


def test_pyautolens_backend_needs_custom_instrument_or_autolens(tmp_path):
    card = LensCard(n_images=1, substructure="none", backend="pyautolens")  # euclid instrument
    result = simulate_lens_batch(card, tmp_path, run_id="p")
    assert result.status == "failed" and ("autolens" in result.message or "custom" in result.message)
