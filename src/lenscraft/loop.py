"""Deterministic active-learning driver (DESIGN.md Section 8) that needs no LLM.

One round = sample_uncertainty -> propose card -> human approval -> simulate -> retrain with the
enlarged training set -> evaluate on the fixed held-out runs. The agent does the same thing
through its tools; this driver exists so the loop can be demonstrated, tested offline, and used
as a scripted arm in the benchmark.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import ToolApproved

from lenscraft.agent.approval import Approver
from lenscraft.compute.backend import ComputeBackend
from lenscraft.ml.uncertainty import sample_uncertainty
from lenscraft.schema.lens_card import LensCard
from lenscraft.schema.ml import EvalSpec, TrainSpec


class LoopRound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round: int
    model_id: str
    weakest: dict[str, Any] | None = None
    proposed_run_id: str | None = None
    proposed_card: LensCard | None = None
    approved: bool = False
    new_run_id: str | None = None
    n_new_images: int = 0
    new_model_id: str | None = None
    metrics_before: dict[str, float] = Field(default_factory=dict)
    metrics_after: dict[str, float] = Field(default_factory=dict)
    message: str = ""


def _eval_metrics(backend: ComputeBackend, model_id: str, test_runs: list[str]) -> dict[str, float]:
    ev = backend.evaluate(EvalSpec(model_id=model_id, run_ids=test_runs))
    if ev.status != "ok":
        return {"error": float("nan")}
    out = dict(ev.metrics)
    if ev.auc is not None:
        out["auc"] = ev.auc
    return out


def run_loop(
    backend: ComputeBackend,
    *,
    model_id: str,
    train_runs: list[str],
    test_runs: list[str],
    rounds: int,
    approve: Approver,
    epochs: int = 10,
    n_images: int = 2000,
    prefix: str = "al",
    max_images: int | None = None,
    input_size: int = 150,
    progress: Callable[[str], None] | None = None,
) -> list[LoopRound]:
    """Run ``rounds`` active-learning rounds starting from an existing trained ``model_id``.

    ``progress`` receives one line per stage (default: stderr) so a long round is not silent.
    """
    say = progress or (lambda m: print(m, file=sys.stderr, flush=True))
    history: list[LoopRound] = []
    current_model = model_id
    train_runs = list(train_runs)
    say(f"evaluating {current_model} on {', '.join(test_runs)} ...")
    metrics = _eval_metrics(backend, current_model, test_runs)
    say(f"  before: {_fmt(metrics)}")

    for r in range(1, rounds + 1):
        entry = LoopRound(round=r, model_id=current_model, metrics_before=metrics)
        say(f"\nround {r}: where is {current_model} weakest?")
        report = sample_uncertainty(backend, current_model, test_runs, n_images_proposed=n_images, seed=1000 + r)
        if report.status != "ok" or report.suggested_card is None:
            entry.message = f"sample_uncertainty failed: {report.message}"
            history.append(entry)
            break
        entry.weakest = report.weakest.model_dump() if report.weakest else None
        if report.weakest:
            w = report.weakest
            say(f"  weakest cell: {w.substructure} / {w.axis} in [{w.low:.3g}, {w.high:.3g}]  (n={w.n}, score {w.score:.2f})")
        run_id = f"{prefix}_r{r}"
        args = {"card": report.suggested_card.model_dump(), "run_id": run_id}
        entry.proposed_run_id, entry.proposed_card = run_id, report.suggested_card

        decision = approve("simulate_lens_batch", args)
        if not isinstance(decision, ToolApproved):
            entry.message = f"denied: {getattr(decision, 'message', '')}"
            history.append(entry)
            break
        if decision.override_args:
            args = decision.override_args
        entry.approved = True
        card = LensCard.model_validate(args["card"])
        run_id = str(args["run_id"])

        say(f"  approved: simulating {card.n_images} {card.substructure} images as {run_id} ...")
        st = backend.wait(backend.submit(card, run_id))
        if st.state != "done" or st.result is None:
            entry.message = f"simulation failed: {st.error}"
            history.append(entry)
            break
        entry.new_run_id, entry.n_new_images = run_id, st.result.n_images
        say(f"  {st.result.n_images} images rendered")
        train_runs.append(run_id)

        base = backend.read_eval_scores(model_id, test_runs) or {}
        kind = base.get("kind", "classifier")
        arch = _model_arch(backend, current_model, kind)
        new_model = f"{model_id}_r{r}"
        spec = TrainSpec(model_id=new_model, kind=kind, arch=arch, run_ids=train_runs, epochs=epochs, max_images=max_images, input_size=input_size, seed=r)
        say(f"  retraining {arch} as {new_model} on {len(train_runs)} runs, {epochs} epochs ...")
        st = backend.wait(backend.submit_train(spec))
        if st.state != "done":
            entry.message = f"training failed: {st.error}"
            history.append(entry)
            break
        entry.new_model_id = new_model
        say(f"  evaluating {new_model} on the same held-out runs ...")
        metrics = _eval_metrics(backend, new_model, test_runs)
        entry.metrics_after = metrics
        say(f"  before: {_fmt(entry.metrics_before)}\n  after:  {_fmt(metrics)}")
        history.append(entry)
        current_model = new_model
    return history


def _fmt(m: dict[str, float]) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in m.items())


def _model_arch(backend: ComputeBackend, model_id: str, kind: str) -> str:
    """Best-effort: reuse the base model's arch if the backend exposes its train result."""
    getter = getattr(backend, "read_train_result", None)
    if getter is not None:
        res = getter(model_id)
        if res is not None and res.get("arch"):
            return str(res["arch"])
    return "resnet18" if kind == "classifier" else "aae"
