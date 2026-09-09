"""Compute backends behind one ``submit()/status()`` interface (DESIGN.md Section 10).

- :class:`LocalBackend` runs simulations in-process and stores runs under a local ``data/`` dir.
- :class:`ModalBackend` submits to the deployed Modal app (``lenscraft.compute.modal_app``);
  images and records live on the ``lenscraft-data`` Modal Volume and never touch the laptop.

The agent's tools only ever talk to :class:`ComputeBackend`, so swapping backends is a CLI flag.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from lenscraft.schema.lens_card import LensCard, SimBatchResult
from lenscraft.schema.lensjsonl import LensRecord, read_records

JobState = Literal["queued", "running", "done", "failed"]


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState
    result: SimBatchResult | None = None
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")


@runtime_checkable
class ComputeBackend(Protocol):
    name: str

    def submit(self, card: LensCard, run_id: str) -> str:
        """Start a simulation batch; returns a job id."""

    def status(self, job_id: str) -> JobStatus: ...

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus: ...

    def list_runs(self) -> list[str]: ...

    def read_records(self, run_id: str) -> list[LensRecord]: ...


def _wait(backend: ComputeBackend, job_id: str, poll_seconds: float, timeout: float | None) -> JobStatus:
    t0 = time.monotonic()
    while True:
        st = backend.status(job_id)
        if st.finished:
            return st
        if timeout is not None and time.monotonic() - t0 > timeout:
            return JobStatus(job_id=job_id, state="failed", error=f"timed out after {timeout:.0f}s waiting for {job_id}")
        time.sleep(poll_seconds)


# ------------------------------------------------------------------------------------------
# Local
# ------------------------------------------------------------------------------------------


class LocalBackend:
    """Runs synchronously in-process. ``submit`` returns only once the batch is finished."""

    name = "local"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self._jobs: dict[str, JobStatus] = {}

    def submit(self, card: LensCard, run_id: str) -> str:
        from lenscraft.sim.batch import simulate_lens_batch

        job_id = f"local-{len(self._jobs) + 1}-{run_id}"
        result = simulate_lens_batch(card, self.data_dir, run_id)
        self._jobs[job_id] = JobStatus(
            job_id=job_id,
            state="done" if result.status == "ok" else "failed",
            result=result,
            error=result.message,
        )
        return job_id

    def status(self, job_id: str) -> JobStatus:
        try:
            return self._jobs[job_id]
        except KeyError:
            return JobStatus(job_id=job_id, state="failed", error=f"unknown job {job_id!r}")

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus:
        return _wait(self, job_id, poll_seconds, timeout)

    def list_runs(self) -> list[str]:
        if not self.data_dir.exists():
            return []
        return sorted(p.name for p in self.data_dir.iterdir() if (p / "records.lensjsonl").exists())

    def read_records(self, run_id: str) -> list[LensRecord]:
        path = self.data_dir / run_id / "records.lensjsonl"
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} under {self.data_dir}")
        return read_records(path)


# ------------------------------------------------------------------------------------------
# Modal
# ------------------------------------------------------------------------------------------


class ModalBackend:
    """Talks to the deployed Modal app. Deploy once with ``modal deploy -m lenscraft.compute.modal_app``.

    ``functions`` and ``call_from_id`` are injectable so the backend can be unit-tested without
    a Modal account; by default they resolve to ``modal.Function.from_name`` / ``modal.FunctionCall.from_id``.
    """

    name = "modal"

    def __init__(
        self,
        app_name: str = "lenscraft",
        *,
        functions: dict[str, Any] | None = None,
        call_from_id: Callable[[str], Any] | None = None,
    ) -> None:
        self.app_name = app_name
        self._functions: dict[str, Any] = dict(functions or {})
        self._call_from_id = call_from_id

    def _fn(self, name: str) -> Any:
        if name not in self._functions:
            import modal

            self._functions[name] = modal.Function.from_name(self.app_name, name)
        return self._functions[name]

    def _call(self, job_id: str) -> Any:
        if self._call_from_id is not None:
            return self._call_from_id(job_id)
        import modal

        return modal.FunctionCall.from_id(job_id)

    def submit(self, card: LensCard, run_id: str) -> str:
        call = self._fn("simulate_batch").spawn(card.model_dump_json(), run_id)
        return str(call.object_id)

    def status(self, job_id: str) -> JobStatus:
        call = self._call(job_id)
        try:
            payload = call.get(timeout=0)
        except TimeoutError:
            return JobStatus(job_id=job_id, state="running")
        except Exception as exc:  # noqa: BLE001 - remote exception or expired call
            return JobStatus(job_id=job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
        result = SimBatchResult.model_validate(payload)
        return JobStatus(
            job_id=job_id,
            state="done" if result.status == "ok" else "failed",
            result=result,
            error=result.message,
        )

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus:
        return _wait(self, job_id, poll_seconds, timeout)

    def list_runs(self) -> list[str]:
        return list(self._fn("list_runs").remote())

    def read_records(self, run_id: str) -> list[LensRecord]:
        text: str = self._fn("read_run").remote(run_id)
        return [LensRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]


def get_backend(name: str, *, data_dir: str | Path = "data") -> ComputeBackend:
    if name == "local":
        return LocalBackend(data_dir)
    if name == "modal":
        return ModalBackend()
    raise ValueError(f"unknown compute backend {name!r}; expected 'local' or 'modal'")
