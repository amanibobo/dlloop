"""sample_uncertainty analysis on synthetic scores, and the closed loop on synthetic data."""

import numpy as np
import pytest

from lenscraft.ml.uncertainty import analyze, per_image_uncertainty
from lenscraft.schema import LensCard, LensRecord, MomentStats


def _rec(run, i, cls, mf=0.03, snr=100.0, extras=None):
    return LensRecord(
        image_id=f"{run}_{i:05d}", lens_card=LensCard(n_images=10, substructure=cls, substructure_mass_fraction=mf or 0.01, seed=1),
        substructure_type=cls, mass_fraction=mf, snr=snr, image_path=f"images/{i:05d}.npy",
        moment_stats=MomentStats(residual_rms=0, second_moment=0.5, arc_ellipticity=0.1), extras=extras or {},
    )


def test_classifier_uncertainty_and_proposal_targets_weak_axion_masses():
    rng = np.random.default_rng(0)
    records, images = [], []
    # vortex images: light axions (long vortices) are easy, heavy ones (short) are hard
    for i in range(40):
        am = 10 ** rng.uniform(-24, -22)
        hard = am > 1e-23
        p_true = 0.4 if hard else 0.95
        probs = [(1 - p_true) / 2, (1 - p_true) / 2, p_true]
        records.append(_rec("vor", i, "vortex", extras={"axion_mass": am}))
        images.append({"image_id": f"vor_{i:05d}", "label": "vortex", "probs": probs})
    for i in range(20):
        records.append(_rec("none", i, "none"))
        images.append({"image_id": f"none_{i:05d}", "label": "none", "probs": [0.99, 0.005, 0.005]})

    report = analyze("clf", "classifier", ["vor", "none"], {"images": images}, records, n_bins=2, min_cell_n=5, n_images_proposed=500, seed=7)
    assert report.status == "ok" and report.n_images == 60
    assert report.per_class_uncertainty["none"] < report.per_class_uncertainty["vortex"]
    w = report.weakest
    assert w.substructure == "vortex" and w.axis == "axion_mass" and w.low >= 1e-23 * 0.5
    assert report.cells[0] == w and all(c.n >= 5 for c in report.cells)
    card = report.suggested_card
    assert card.substructure == "vortex" and card.n_images == 500 and card.seed == 7
    assert w.low <= card.axion_mass <= w.high
    assert "axion_mass" in report.rationale


def test_anomaly_uncertainty():
    none_scores = np.linspace(0.01, 0.10, 50)
    images = [{"image_id": f"n_{i:05d}", "label": "none", "score": float(s)} for i, s in enumerate(none_scores)]
    images += [{"image_id": "v_00000", "label": "vortex", "score": 0.5}, {"image_id": "v_00001", "label": "vortex", "score": 0.05}]
    u, ok = per_image_uncertainty("anomaly", images)
    assert ok[-2] == 1 and ok[-1] == 0  # loud anomaly detected, quiet one missed
    assert u[-2] < 0.05 and u[-1] > 0.4  # quiet anomaly looks ordinary -> uncertain
    assert u[0] < u[49]  # among 'none', the highest-scoring is the most outlying

    records = [_rec("n", i, "none", mf=0.0) for i in range(50)] + [_rec("v", 0, "vortex"), _rec("v", 1, "vortex")]
    report = analyze("aae", "anomaly", ["n", "v"], {"images": images}, records, n_bins=2, min_cell_n=1)
    assert report.kind == "anomaly" and report.weakest is not None
    assert "detection rate" in report.rationale


def test_snr_axis_proposal_keeps_base_card():
    records = [_rec("s", i, "subhalo", snr=float(50 + i * 10), extras={"n_subhalos": 20.0}) for i in range(20)]
    images = [{"image_id": f"s_{i:05d}", "label": "subhalo", "probs": [0.6, 0.4, 0.0] if i < 10 else [0.05, 0.95, 0.0]} for i in range(20)]
    report = analyze("clf", "classifier", ["s"], {"images": images}, records, n_bins=2, min_cell_n=5)
    assert report.weakest.axis == "snr" and report.weakest.high <= 150
    assert report.suggested_card.substructure == "subhalo" and "SNR is not a LensCard field" in report.rationale


def test_no_matching_images_is_structured():
    report = analyze("m", "classifier", ["x"], {"images": [{"image_id": "zzz", "label": "none", "probs": [1, 0, 0]}]}, [])
    assert report.status == "failed"


@pytest.mark.integration
def test_closed_loop_on_synthetic_data(synthetic_dataset):
    pytest.importorskip("torch")
    pytest.importorskip("lenstronomy")
    from lenscraft.agent import auto_approver
    from lenscraft.compute import LocalBackend
    from lenscraft.loop import run_loop
    from lenscraft.schema import TrainSpec

    backend = LocalBackend(synthetic_dataset)
    backend.submit_train(TrainSpec(model_id="clf0", kind="classifier", arch="resnet18", run_ids=["none_a", "sub_a", "vor_a"], epochs=1, input_size=96, batch_size=8))

    seen = []

    def approve(tool_name, args):
        seen.append((tool_name, args["run_id"]))
        # keep the real simulation tiny and inside the Euclid field of view
        card = {**args["card"], "n_images": 2, "image_size": 32}
        from pydantic_ai import ToolApproved

        return ToolApproved(override_args={"card": card, "run_id": args["run_id"]})

    history = run_loop(backend, model_id="clf0", train_runs=["none_a", "sub_a", "vor_a"], test_runs=["none_b", "vor_b"], rounds=1,
                       approve=approve, epochs=1, n_images=2, input_size=96, prefix="t")
    assert seen == [("simulate_lens_batch", "t_r1")]
    h = history[0]
    assert h.approved and h.new_run_id == "t_r1" and h.n_new_images == 2 and h.new_model_id == "clf0_r1", h.message
    assert "accuracy" in h.metrics_before and "accuracy" in h.metrics_after
    assert backend.list_models() == ["clf0", "clf0_r1"] and "t_r1" in backend.list_runs()
    assert (synthetic_dataset / "_models" / "clf0" / "eval_none_b-vor_b.json").exists()
