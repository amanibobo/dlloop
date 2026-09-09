"""Tiny CPU training runs on synthetic data through LocalBackend (no Lenstronomy, no GPU)."""

import json

import pytest

torch = pytest.importorskip("torch")

from lenscraft.compute import LocalBackend  # noqa: E402
from lenscraft.schema import EvalSpec, TrainSpec  # noqa: E402


def test_train_spec_validation():
    with pytest.raises(ValueError):
        TrainSpec(model_id="m", kind="classifier", arch="aae", run_ids=["a"])
    with pytest.raises(ValueError):
        TrainSpec(model_id="bad id", kind="anomaly", arch="aae", run_ids=["a"])
    assert TrainSpec(model_id="m", kind="anomaly", arch="vae", run_ids=["a"]).input_size == 150


def test_classifier_train_and_evaluate(synthetic_dataset):
    backend = LocalBackend(synthetic_dataset)
    spec = TrainSpec(model_id="clf", kind="classifier", arch="resnet18", run_ids=["none_a", "sub_a", "vor_a"], epochs=2, input_size=96, batch_size=8)
    st = backend.status(backend.submit_train(spec))
    assert st.state == "done", st.error
    r = st.result
    assert r.n_train + r.n_val == 36 and r.epochs_run == 2 and len(r.history) == 2
    assert {"train_loss", "val_loss", "val_acc"} <= set(r.metrics)
    assert backend.list_models() == ["clf"]
    assert (synthetic_dataset / "_models" / "clf" / "model.pt").exists()

    ev = backend.evaluate(EvalSpec(model_id="clf", run_ids=["none_b", "vor_b"]))
    assert ev.status == "ok", ev.message
    assert ev.kind == "classifier" and ev.n_images == 12
    assert set(ev.per_class) == {"none", "vortex"} and 0 <= ev.metrics["accuracy"] <= 1
    assert ev.auc is not None and 0 <= ev.auc <= 1
    scores = json.loads(open(ev.scores_path).read())
    assert len(scores["images"]) == 12 and "probs" in scores["images"][0]


@pytest.mark.parametrize("arch", ["dcae", "vae", "aae"])
def test_anomaly_train_and_evaluate(synthetic_dataset, arch):
    backend = LocalBackend(synthetic_dataset)
    spec = TrainSpec(model_id=f"an_{arch}", kind="anomaly", arch=arch, run_ids=["none_a", "vor_a"], epochs=2, input_size=96, latent_dim=32, batch_size=8)
    st = backend.status(backend.submit_train(spec))
    assert st.state == "done", st.error
    r = st.result
    assert r.n_train + r.n_val == 12  # only the 'none' class is trained on
    assert "auc" in r.metrics  # vortex images held out and scored

    ev = backend.evaluate(EvalSpec(model_id=f"an_{arch}", run_ids=["none_b", "vor_b"]))
    assert ev.status == "ok", ev.message
    assert ev.kind == "anomaly" and ev.auc is not None
    assert set(ev.per_class) == {"none", "vortex"} and "mean_score" in ev.per_class["vortex"]


def test_anomaly_requires_none_class(synthetic_dataset):
    backend = LocalBackend(synthetic_dataset)
    spec = TrainSpec(model_id="bad", kind="anomaly", arch="dcae", run_ids=["vor_a"], epochs=1, input_size=96, latent_dim=8)
    st = backend.status(backend.submit_train(spec))
    assert st.state == "failed" and "none" in st.error


def test_failures_are_structured(synthetic_dataset):
    backend = LocalBackend(synthetic_dataset)
    st = backend.status(backend.submit_train(TrainSpec(model_id="x", kind="classifier", arch="resnet18", run_ids=["missing"], epochs=1)))
    assert st.state == "failed" and "missing" in st.error
    ev = backend.evaluate(EvalSpec(model_id="nope", run_ids=["none_b"]))
    assert ev.status == "failed" and "nope" in ev.message
