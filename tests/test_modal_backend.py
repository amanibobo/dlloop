"""ModalBackend logic with fake Modal handles: no account or network needed."""

from lenscraft.compute import ModalBackend
from lenscraft.schema import EvalSpec, LensCard, TrainSpec


class FakeCall:
    def __init__(self, object_id, payload=None, error=None, pending=False):
        self.object_id, self._payload, self._error, self._pending = object_id, payload, error, pending

    def get(self, timeout=None):
        if self._pending:
            raise TimeoutError("not ready")
        if self._error:
            raise RuntimeError(self._error)
        return self._payload


class FakeSpawn:
    def __init__(self, prefix):
        self.prefix, self.calls = prefix, []

    def spawn(self, *args):
        self.calls.append(args)
        return FakeCall(f"{self.prefix}-{len(self.calls)}")


class FakeRemote:
    def __init__(self, value):
        self.value, self.calls = value, []

    def remote(self, *args):
        self.calls.append(args)
        return self.value


def _sim_payload(run_id, status="ok", n=3):
    return {
        "status": status, "run_id": run_id, "lensjsonl_path": f"/data/{run_id}/records.lensjsonl",
        "image_dir": f"/data/{run_id}/images", "n_images": n, "elapsed_seconds": 1.5,
        "message": "" if status == "ok" else "boom",
    }


def _train_payload(model_id, status="ok"):
    return {"status": status, "model_id": model_id, "kind": "classifier", "arch": "resnet18", "checkpoint_path": f"/data/_models/{model_id}/model.pt",
            "n_train": 80, "n_val": 20, "epochs_run": 2, "metrics": {"val_acc": 0.9}, "history": [], "elapsed_seconds": 9.0, "message": "" if status == "ok" else "oom"}


def test_submit_and_status_transitions():
    sim = FakeSpawn("fc")
    calls = {}
    backend = ModalBackend(functions={"simulate_batch": sim}, call_from_id=lambda jid: calls[jid])
    card = LensCard(n_images=3, substructure="vortex", seed=0)
    job = backend.submit(card, "v1")
    assert job == "fc-1" and sim.calls[0][1] == "v1"
    assert LensCard.model_validate_json(sim.calls[0][0]) == card

    calls[job] = FakeCall(job, pending=True)
    assert backend.status(job).state == "running"
    calls[job] = FakeCall(job, payload=_sim_payload("v1"))
    st = backend.status(job)
    assert st.state == "done" and st.result.n_images == 3
    calls[job] = FakeCall(job, payload=_sim_payload("v1", status="failed"))
    assert backend.status(job).state == "failed" and backend.status(job).error == "boom"
    calls[job] = FakeCall(job, error="container died")
    assert "container died" in backend.status(job).error


def test_train_submit_status_and_evaluate():
    train = FakeSpawn("tr")
    calls = {}
    ev = FakeRemote({"status": "ok", "model_id": "clf", "kind": "classifier", "n_images": 10, "auc": 0.95, "metrics": {}, "per_class": {}, "scores_path": "/data/_models/clf/eval_x.json", "elapsed_seconds": 1.0, "message": ""})
    backend = ModalBackend(functions={"train_model": train, "evaluate_model": ev, "list_models": FakeRemote(["clf"])}, call_from_id=lambda jid: calls[jid])
    spec = TrainSpec(model_id="clf", kind="classifier", arch="resnet18", run_ids=["a", "b"], epochs=2)
    job = backend.submit_train(spec)
    assert TrainSpec.model_validate_json(train.calls[0][0]) == spec
    calls[job] = FakeCall(job, payload=_train_payload("clf"))
    st = backend.status(job)
    assert st.state == "done" and st.result.metrics["val_acc"] == 0.9 and st.result.model_id == "clf"
    calls[job] = FakeCall(job, payload=_train_payload("clf", status="failed"))
    assert backend.status(job).error == "oom"

    res = backend.evaluate(EvalSpec(model_id="clf", run_ids=["c"]))
    assert res.auc == 0.95 and EvalSpec.model_validate_json(ev.calls[0][0]).run_ids == ["c"]
    assert backend.list_models() == ["clf"]


def test_wait_times_out():
    calls = {"j": FakeCall("j", pending=True)}
    backend = ModalBackend(functions={}, call_from_id=lambda jid: calls[jid])
    st = backend.wait("j", poll_seconds=0.01, timeout=0.05)
    assert st.state == "failed" and "timed out" in st.error


def test_read_records_and_list_runs():
    from lenscraft.schema import LensRecord, MomentStats

    rec = LensRecord(
        image_id="v1_00000", lens_card=LensCard(n_images=1, substructure="none"), substructure_type="none",
        mass_fraction=0.0, snr=10.0, image_path="images/00000.npy",
        moment_stats=MomentStats(residual_rms=0, second_moment=0.5, arc_ellipticity=0.1),
    )
    backend = ModalBackend(functions={"read_run": FakeRemote(rec.model_dump_json() + "\n"), "list_runs": FakeRemote(["v1"])})
    assert backend.list_runs() == ["v1"]
    assert backend.read_records("v1") == [rec]
