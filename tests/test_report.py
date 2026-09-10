import json

import pytest

from lenscraft.compute import LocalBackend
from lenscraft.report import generate_report


def test_report_dataset_only(synthetic_dataset, tmp_path):
    backend = LocalBackend(synthetic_dataset)
    res = generate_report(backend, run_ids=["none_a", "vor_a"], out_dir=tmp_path / "r", title="T")
    assert res.status == "ok", res.message
    md = (tmp_path / "r" / "report.md").read_text()
    assert md.startswith("# T") and "| `none_a` | none | 12 |" in md
    assert set(res.figures) == {"dataset_distributions.png", "sample_images.png"}
    for f in res.figures:
        assert (tmp_path / "r" / f).stat().st_size > 1000


def test_report_with_models_loop_and_bench(synthetic_dataset, tmp_path):
    pytest.importorskip("torch")
    from lenscraft.schema import EvalSpec, TrainSpec

    backend = LocalBackend(synthetic_dataset)
    backend.submit_train(TrainSpec(model_id="clf", kind="classifier", arch="resnet18", run_ids=["none_a", "sub_a", "vor_a"], epochs=1, input_size=96, batch_size=8))
    backend.submit_train(TrainSpec(model_id="an", kind="anomaly", arch="dcae", run_ids=["none_a", "vor_a"], epochs=1, input_size=96, latent_dim=8, batch_size=8))
    backend.evaluate(EvalSpec(model_id="clf", run_ids=["none_b", "vor_b"]))
    backend.evaluate(EvalSpec(model_id="an", run_ids=["none_b", "vor_b"]))
    loop = [{"round": 1, "model_id": "clf", "weakest": {"substructure": "vortex", "axis": "snr", "low": 90, "high": 110, "n": 6, "mean_uncertainty": 0.5, "score": 0.5},
             "new_run_id": "al_r1", "n_new_images": 2, "new_model_id": "clf_r1", "metrics_before": {"accuracy": 0.5, "auc": 0.6}, "metrics_after": {"accuracy": 0.7, "auc": 0.8}}]
    bench = {"arms": [
        {"arm": "tool", "n": 3, "successes": 3, "pass_at_k": {"1": 1.0, "3": 1.0}, "reach_mean": 1.0, "reach_std": 0.0, "stage_pass_rate": {"simulate": 1, "label": 1, "train": 1, "auc": 1}, "mean_requests": 9, "mean_tokens": 5000, "mean_duration_seconds": 60},
        {"arm": "core", "n": 3, "successes": 0, "pass_at_k": {"1": 0.0, "3": 0.0}, "reach_mean": 0.2, "reach_std": 0.1, "stage_pass_rate": {"simulate": 0.33, "label": 0.33, "train": 0, "auc": 0}, "mean_requests": 30, "mean_tokens": 90000, "mean_duration_seconds": 600},
    ]}
    res = generate_report(backend, run_ids=["none_a", "sub_a", "vor_a", "none_b", "vor_b"], model_ids=["clf", "an"], eval_runs=["none_b", "vor_b"],
                          out_dir=tmp_path / "r", loop_history=loop, bench_summary=bench)
    assert res.status == "ok", res.message
    md = (tmp_path / "r" / "report.md").read_text()
    assert "## Model `clf`" in md and "| true \\ predicted |" in md
    assert "## Model `an`" in md and "scores_an.png" in md
    assert "Where clf is weakest" not in md and "uncertainty_clf.png" in md
    assert "## Active-learning rounds" in md and "`al_r1`" in md
    assert "## Benchmark" in md and "bench_stages.png" in md
    assert {"train_clf.png", "val_clf.png", "uncertainty_clf.png", "bench_stages.png"} <= set(res.figures)


def test_report_failure_is_structured(tmp_path):
    res = generate_report(LocalBackend(tmp_path / "nothing"), run_ids=["missing"], out_dir=tmp_path / "r")
    assert res.status == "failed" and "missing" in res.message
    assert json.loads(res.model_dump_json())["status"] == "failed"
