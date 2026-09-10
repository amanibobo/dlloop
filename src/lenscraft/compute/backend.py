"""Compute backends behind one ``submit()/status()`` interface (DESIGN.md Section 10).

- :class:`LocalBackend` runs simulations and training in-process; runs live under ``<data_dir>``
  and models under ``<data_dir>/_models``.
- :class:`ModalBackend` submits to the deployed Modal app (``lenscraft.compute.modal_app``);
  images, records and checkpoints live on the ``lenscraft-data`` Volume and never touch the laptop.

The agent's tools only ever talk to :class:`ComputeBackend`, so swapping backends is a CLI flag.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from lenscraft.schema.lens_card import LensCard, SimBatchResult
from lenscraft.schema.lensjsonl import LensRecord, read_records
from lenscraft.schema.ml import EvalResult, EvalSpec, TrainResult, TrainSpec, eval_scores_filename

JobState = Literal["queued", "running", "done", "failed"]
MODELS_DIRNAME = "_models"


def eval_scores_relpath(model_id: str, run_ids: list[str]) -> str:
    """Path of a scores file relative to the data root (same layout locally and on the Volume)."""
    return f"{MODELS_DIRNAME}/{model_id}/{eval_scores_filename(run_ids)}"


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState
    result: SimBatchResult | TrainResult | None = None
    error: str = ""

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")


@runtime_checkable
class ComputeBackend(Protocol):
    name: str

    def submit(self, card: LensCard, run_id: str) -> str:
        """Start a simulation batch; returns a job id."""

    def submit_train(self, spec: TrainSpec) -> str:
        """Start a training job; returns a job id."""

    def status(self, job_id: str) -> JobStatus: ...

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus: ...

    def evaluate(self, spec: EvalSpec) -> EvalResult: ...

    def list_runs(self) -> list[str]: ...

    def list_models(self) -> list[str]: ...

    def read_records(self, run_id: str) -> list[LensRecord]: ...

    def read_eval_scores(self, model_id: str, run_ids: list[str]) -> dict[str, Any] | None:
        """Per-image scores written by ``evaluate`` for these runs, or None if not evaluated yet."""


def _wait(backend: ComputeBackend, job_id: str, poll_seconds: float, timeout: float | None) -> JobStatus:
    t0 = time.monotonic()
    while True:
        st = backend.status(job_id)
        if st.finished:
            return st
        if timeout is not None and time.monotonic() - t0 > timeout:
            return JobStatus(job_id=job_id, state="failed", error=f"timed out after {timeout:.0f}s waiting for {job_id}")
        time.sleep(poll_seconds)


def _status_from_result(job_id: str, result: SimBatchResult | TrainResult) -> JobStatus:
    return JobStatus(job_id=job_id, state="done" if result.status == "ok" else "failed", result=result, error=result.message)


# ------------------------------------------------------------------------------------------
# Local
# ------------------------------------------------------------------------------------------


class LocalBackend:
    """Runs synchronously in-process. ``submit*`` return only once the job is finished."""

    name = "local"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / MODELS_DIRNAME
        self._jobs: dict[str, JobStatus] = {}

    def submit(self, card: LensCard, run_id: str) -> str:
        from lenscraft.sim.batch import simulate_lens_batch

        job_id = f"local-{len(self._jobs) + 1}-{run_id}"
        self._jobs[job_id] = _status_from_result(job_id, simulate_lens_batch(card, self.data_dir, run_id))
        return job_id

    def submit_train(self, spec: TrainSpec) -> str:
        from lenscraft.ml.train import train_model  # needs torch

        job_id = f"local-{len(self._jobs) + 1}-{spec.model_id}"
        self._jobs[job_id] = _status_from_result(job_id, train_model(spec, self.data_dir, self.models_dir))
        return job_id

    def status(self, job_id: str) -> JobStatus:
        try:
            return self._jobs[job_id]
        except KeyError:
            return JobStatus(job_id=job_id, state="failed", error=f"unknown job {job_id!r}")

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus:
        return _wait(self, job_id, poll_seconds, timeout)

    def evaluate(self, spec: EvalSpec) -> EvalResult:
        from lenscraft.ml.train import evaluate_model  # needs torch

        return evaluate_model(spec, self.data_dir, self.models_dir)

    def list_runs(self) -> list[str]:
        if not self.data_dir.exists():
            return []
        return sorted(p.name for p in self.data_dir.iterdir() if (p / "records.lensjsonl").exists())

    def list_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(p.name for p in self.models_dir.iterdir() if (p / "model.pt").exists())

    def read_records(self, run_id: str) -> list[LensRecord]:
        path = self.data_dir / run_id / "records.lensjsonl"
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} under {self.data_dir}")
        return read_records(path)

    def read_eval_scores(self, model_id: str, run_ids: list[str]) -> dict[str, Any] | None:
        path = self.data_dir / eval_scores_relpath(model_id, run_ids)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_train_result(self, model_id: str) -> dict[str, Any] | None:
        path = self.models_dir / model_id / "train_result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_image(self, run_id: str, image_path: str) -> np.ndarray | None:
        path = self.data_dir / run_id / image_path
        return np.load(path) if path.exists() else None


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
        self._job_kind: dict[str, str] = {}

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
        self._job_kind[str(call.object_id)] = "sim"
        return str(call.object_id)

    def submit_train(self, spec: TrainSpec) -> str:
        call = self._fn("train_model").spawn(spec.model_dump_json())
        self._job_kind[str(call.object_id)] = "train"
        return str(call.object_id)

    def status(self, job_id: str) -> JobStatus:
        call = self._call(job_id)
        try:
            payload = call.get(timeout=0)
        except TimeoutError:
            return JobStatus(job_id=job_id, state="running")
        except Exception as exc:  # noqa: BLE001 - remote exception or expired call
            return JobStatus(job_id=job_id, state="failed", error=f"{type(exc).__name__}: {exc}")
        kind = self._job_kind.get(job_id) or ("train" if "model_id" in payload else "sim")
        result = TrainResult.model_validate(payload) if kind == "train" else SimBatchResult.model_validate(payload)
        return _status_from_result(job_id, result)

    def wait(self, job_id: str, *, poll_seconds: float = 2.0, timeout: float | None = None) -> JobStatus:
        return _wait(self, job_id, poll_seconds, timeout)

    def evaluate(self, spec: EvalSpec) -> EvalResult:
        return EvalResult.model_validate(self._fn("evaluate_model").remote(spec.model_dump_json()))

    def list_runs(self) -> list[str]:
        return list(self._fn("list_runs").remote())

    def list_models(self) -> list[str]:
        return list(self._fn("list_models").remote())

    def read_records(self, run_id: str) -> list[LensRecord]:
        text: str = self._fn("read_run").remote(run_id)
        return [LensRecord.model_validate_json(line) for line in text.splitlines() if line.strip()]

    def read_eval_scores(self, model_id: str, run_ids: list[str]) -> dict[str, Any] | None:
        try:
            text: str = self._fn("read_file").remote(eval_scores_relpath(model_id, run_ids))
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - Modal wraps remote exceptions
            if "FileNotFound" in type(exc).__name__ or "no file" in str(exc):
                return None
            raise
        return json.loads(text)

    def read_train_result(self, model_id: str) -> dict[str, Any] | None:
        try:
            text: str = self._fn("read_file").remote(f"{MODELS_DIRNAME}/{model_id}/train_result.json")
        except Exception as exc:  # noqa: BLE001
            if "FileNotFound" in type(exc).__name__ or "no file" in str(exc):
                return None
            raise
        return json.loads(text)

    def read_image(self, run_id: str, image_path: str) -> np.ndarray | None:
        import io

        try:
            data: bytes = self._fn("read_npy").remote(f"{run_id}/{image_path}")
        except Exception as exc:  # noqa: BLE001
            if "FileNotFound" in type(exc).__name__ or "no file" in str(exc):
                return None
            raise
        return np.load(io.BytesIO(data))


def get_backend(name: str, *, data_dir: str | Path = "data") -> ComputeBackend:
    if name == "local":
        return LocalBackend(data_dir)
    if name == "modal":
        return ModalBackend()
    raise ValueError(f"unknown compute backend {name!r}; expected 'local' or 'modal'")
