"""ModalBackend logic with fake Modal handles: no account or network needed."""

from lenscraft.compute import ModalBackend
from lenscraft.schema import LensCard


class FakeCall:
    def __init__(self, object_id, payload=None, error=None, pending=False):
        self.object_id, self._payload, self._error, self._pending = object_id, payload, error, pending

    def get(self, timeout=None):
        if self._pending:
            raise TimeoutError("not ready")
        if self._error:
            raise RuntimeError(self._error)
        return self._payload


class FakeSimulate:
    def __init__(self):
        self.calls = []

    def spawn(self, card_json, run_id):
        self.calls.append((card_json, run_id))
        return FakeCall(f"fc-{run_id}")


class FakeRemote:
    def __init__(self, value):
        self.value = value

    def remote(self, *args):
        return self.value


def _payload(run_id, status="ok", n=3):
    return {
        "status": status, "run_id": run_id, "lensjsonl_path": f"/data/{run_id}/records.lensjsonl",
        "image_dir": f"/data/{run_id}/images", "n_images": n, "elapsed_seconds": 1.5,
        "message": "" if status == "ok" else "boom",
    }


def test_submit_and_status_transitions():
    sim = FakeSimulate()
    calls = {}
    backend = ModalBackend(functions={"simulate_batch": sim}, call_from_id=lambda jid: calls[jid])
    card = LensCard(n_images=3, substructure="vortex", seed=0)
    job = backend.submit(card, "v1")
    assert job == "fc-v1" and sim.calls[0][1] == "v1"
    assert LensCard.model_validate_json(sim.calls[0][0]) == card

    calls[job] = FakeCall(job, pending=True)
    assert backend.status(job).state == "running"
    calls[job] = FakeCall(job, payload=_payload("v1"))
    st = backend.status(job)
    assert st.state == "done" and st.result.n_images == 3
    calls[job] = FakeCall(job, payload=_payload("v1", status="failed"))
    assert backend.status(job).state == "failed" and backend.status(job).error == "boom"
    calls[job] = FakeCall(job, error="container died")
    assert "container died" in backend.status(job).error


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
