import pytest

from lenscraft.compute import JobStatus, LocalBackend, ModalBackend, get_backend
from lenscraft.schema import LensCard

lenstronomy = pytest.importorskip("lenstronomy")
pytestmark = pytest.mark.integration


def test_local_backend_round_trip(tmp_path):
    backend = LocalBackend(tmp_path)
    assert backend.list_runs() == []
    card = LensCard(n_images=2, substructure="none", image_size=32, seed=1)
    job = backend.submit(card, "r1")
    st = backend.status(job)
    assert st.state == "done" and st.result is not None and st.result.n_images == 2
    assert backend.wait(job) == st
    assert backend.list_runs() == ["r1"]
    assert len(backend.read_records("r1")) == 2
    with pytest.raises(FileNotFoundError):
        backend.read_records("nope")


def test_local_backend_failure_is_reported(tmp_path):
    backend = LocalBackend(tmp_path)
    bad = LensCard(n_images=1, substructure="none", instrument="custom", image_size=32)  # fov too small
    st = backend.status(backend.submit(bad, "bad"))
    assert st.state == "failed" and "field of view" in st.error
    assert backend.status("missing").state == "failed"


def test_get_backend(tmp_path):
    assert isinstance(get_backend("local", data_dir=tmp_path), LocalBackend)
    assert isinstance(get_backend("modal"), ModalBackend)
    with pytest.raises(ValueError):
        get_backend("slurm")


def test_job_status_finished():
    assert not JobStatus(job_id="x", state="running").finished
    assert JobStatus(job_id="x", state="done").finished
