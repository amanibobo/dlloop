import pytest

from lenscraft.schema import LensCard, LensRecord, MomentStats, read_records, summarize_records, write_records


def _rec(i: int, cls: str, snr: float) -> LensRecord:
    return LensRecord(
        image_id=f"run001_{i:05d}",
        lens_card=LensCard(n_images=3, substructure=cls, seed=1),
        substructure_type=cls,
        mass_fraction=0.0 if cls == "none" else 0.01,
        snr=snr,
        image_path=f"images/{i:05d}.npy",
        moment_stats=MomentStats(residual_rms=0.0, second_moment=0.5, arc_ellipticity=0.2),
        extras={"theta_E": 1.6},
    )


def test_write_read_round_trip(tmp_path):
    recs = [_rec(0, "none", 10.0), _rec(1, "vortex", 20.0), _rec(2, "subhalo", 30.0)]
    path = tmp_path / "nested" / "records.lensjsonl"
    assert write_records(path, recs) == 3
    assert path.read_text().count("\n") == 3
    back = read_records(path)
    assert back == recs


def test_invalid_line_reports_line_number(tmp_path):
    path = tmp_path / "bad.lensjsonl"
    path.write_text('{"image_id": "x"}\n')
    with pytest.raises(ValueError, match="bad.lensjsonl:1"):
        read_records(path)


def test_summarize():
    recs = [_rec(0, "none", 10.0), _rec(1, "vortex", 20.0), _rec(2, "vortex", 40.0)]
    s = summarize_records(recs)
    assert s["n_images"] == 3
    assert s["by_class"]["vortex"]["n"] == 2
    assert s["by_class"]["vortex"]["snr"]["mean"] == 30.0
    assert s["runs"] == ["run001"]
    assert summarize_records([])["n_images"] == 0


def test_moment_stats_bounds():
    with pytest.raises(ValueError):
        MomentStats(residual_rms=-1, second_moment=0, arc_ellipticity=0)
    with pytest.raises(ValueError):
        MomentStats(residual_rms=0, second_moment=0, arc_ellipticity=1.5)
