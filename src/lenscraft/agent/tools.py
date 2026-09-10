"""Domain tools (DESIGN.md Section 6): simulation tier and ML tier. Tools only talk to
``ComputeBackend``, never to the engine or to torch directly.

Approval-gated (cost money or GPU time): ``simulate_lens_batch``, ``train_classifier``,
``train_anomaly_detector``. Free: ``summarize_dataset``, ``evaluate_model``, ``list_runs``,
``list_models``.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from lenscraft.agent.deps import AgentDeps
from lenscraft.schema.lens_card import LensCard, SimBatchResult
from lenscraft.schema.lensjsonl import summarize_records
from lenscraft.schema.ml import EvalResult, EvalSpec, TrainResult, TrainSpec
from lenscraft.schema.uncertainty import UncertaintyReport

ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,39}")

APPROVAL_GATED = ("simulate_lens_batch", "train_classifier", "train_anomaly_detector")


def _check_id(value: str, what: str) -> None:
    if not ID_RE.fullmatch(value):
        raise ModelRetry(f"{what} {value!r} is invalid: use 1-40 characters from letters, digits, '_' or '-'")


# --------------------------------------------------------------------------------------------
# Simulation tier
# --------------------------------------------------------------------------------------------


def simulate_lens_batch(ctx: RunContext[AgentDeps], card: LensCard, run_id: str) -> SimBatchResult:
    """Generate a batch of strong-lensing images from a LensCard. Requires human approval.

    Args:
        card: The simulation configuration (substructure class, halo mass, redshifts, instrument, seed, ...).
        run_id: Short unique name for this run, e.g. 'vortex_a'. Letters, digits, '_' or '-' only.
    """
    _check_id(run_id, "run_id")
    if run_id in ctx.deps.backend.list_runs():
        raise ModelRetry(f"run_id {run_id!r} already exists; pick a new one (see list_runs)")

    job_id = ctx.deps.backend.submit(card, run_id)
    status = ctx.deps.backend.wait(job_id, poll_seconds=ctx.deps.poll_seconds, timeout=ctx.deps.wait_timeout)
    if not isinstance(status.result, SimBatchResult):
        return SimBatchResult(
            status="failed", run_id=run_id, lensjsonl_path="", image_dir="", n_images=0,
            elapsed_seconds=0.0, message=status.error or f"job {job_id} ended in state {status.state}",
        )
    if status.result.status == "ok":
        ctx.deps.run_ids.append(run_id)
    return status.result


def summarize_dataset(ctx: RunContext[AgentDeps], run_ids: list[str]) -> dict[str, Any]:
    """Summarise the composition and quality of one or more finished runs.

    Returns per-class counts and min/median/max/mean of SNR, mass fraction and residual_rms.

    Args:
        run_ids: Run ids to include (see list_runs).
    """
    records = []
    for run_id in run_ids:
        try:
            records.extend(ctx.deps.backend.read_records(run_id))
        except FileNotFoundError as exc:
            raise ModelRetry(str(exc)) from exc
    return summarize_records(records)


def list_runs(ctx: RunContext[AgentDeps]) -> list[str]:
    """List the run ids that already exist on the compute backend."""
    return ctx.deps.backend.list_runs()


# --------------------------------------------------------------------------------------------
# ML tier
# --------------------------------------------------------------------------------------------


def _train(ctx: RunContext[AgentDeps], spec: TrainSpec) -> TrainResult:
    if spec.model_id in ctx.deps.backend.list_models():
        raise ModelRetry(f"model_id {spec.model_id!r} already exists; pick a new one (see list_models)")
    existing = set(ctx.deps.backend.list_runs())
    missing = [r for r in spec.run_ids if r not in existing]
    if missing:
        raise ModelRetry(f"unknown run_ids {missing}; see list_runs")
    job_id = ctx.deps.backend.submit_train(spec)
    status = ctx.deps.backend.wait(job_id, poll_seconds=ctx.deps.poll_seconds, timeout=ctx.deps.wait_timeout)
    if not isinstance(status.result, TrainResult):
        return TrainResult(status="failed", model_id=spec.model_id, kind=spec.kind, arch=spec.arch,
                           message=status.error or f"job {job_id} ended in state {status.state}")
    if status.result.status == "ok":
        ctx.deps.model_ids.append(spec.model_id)
    return status.result


def train_classifier(
    ctx: RunContext[AgentDeps],
    model_id: str,
    run_ids: list[str],
    arch: Literal["resnet18", "alexnet"] = "resnet18",
    epochs: int = 10,
    max_images: Optional[int] = None,
    seed: int = 0,
) -> TrainResult:
    """Train a supervised 3-class substructure classifier (none / subhalo / vortex). Requires human approval.

    The runs should together contain at least two classes. Images are resized to 150 px.
    Returns validation accuracy and macro AUC, plus per-epoch history.

    Args:
        model_id: Unique name for the model, e.g. 'clf_resnet_a'.
        run_ids: Runs to train on (see list_runs); a stratified 20% validation split is held out.
        arch: 'resnet18' (default, best in Alexander et al. 2019) or 'alexnet'.
        epochs: Training epochs.
        max_images: Optional cap on the number of images (quick tests).
        seed: Random seed for the split and initialisation.
    """
    _check_id(model_id, "model_id")
    spec = TrainSpec(model_id=model_id, kind="classifier", arch=arch, run_ids=run_ids, epochs=epochs, max_images=max_images, seed=seed)
    return _train(ctx, spec)


def train_anomaly_detector(
    ctx: RunContext[AgentDeps],
    model_id: str,
    run_ids: list[str],
    arch: Literal["aae", "vae", "dcae"] = "aae",
    epochs: int = 10,
    max_images: Optional[int] = None,
    seed: int = 0,
) -> TrainResult:
    """Train an unsupervised anomaly detector on the 'none' class. Requires human approval.

    Only substructure='none' images are used for training; any subhalo/vortex images in the same
    runs are held out and used to report how well reconstruction error separates them (AUC).
    Images are resized to 150 px (the architecture size in Alexander et al. 2021).

    Args:
        model_id: Unique name for the model, e.g. 'aae_a'.
        run_ids: Runs to draw images from (see list_runs). Must include a 'none' run.
        arch: 'aae' (default, best in the 2021 paper), 'vae' or 'dcae'.
        epochs: Training epochs.
        max_images: Optional cap on the number of training images (quick tests).
        seed: Random seed.
    """
    _check_id(model_id, "model_id")
    spec = TrainSpec(model_id=model_id, kind="anomaly", arch=arch, run_ids=run_ids, epochs=epochs, max_images=max_images, seed=seed)
    return _train(ctx, spec)


def evaluate_model(ctx: RunContext[AgentDeps], model_id: str, run_ids: list[str]) -> EvalResult:
    """Evaluate a trained model on one or more runs.

    Classifier: accuracy, macro one-vs-rest AUC, per-class accuracy/AUC/mean confidence.
    Anomaly detector: substructure-vs-none AUC and mean anomaly score per class.
    Per-image scores are written to scores_path for later uncertainty analysis.

    Args:
        model_id: A trained model (see list_models).
        run_ids: Runs to evaluate on (see list_runs).
    """
    if model_id not in ctx.deps.backend.list_models():
        raise ModelRetry(f"unknown model_id {model_id!r}; see list_models")
    return ctx.deps.backend.evaluate(EvalSpec(model_id=model_id, run_ids=run_ids))


def list_models(ctx: RunContext[AgentDeps]) -> list[str]:
    """List the trained model ids on the compute backend."""
    return ctx.deps.backend.list_models()


def sample_uncertainty(
    ctx: RunContext[AgentDeps],
    model_id: str,
    run_ids: list[str],
    n_images_proposed: int = 2000,
    n_bins: int = 4,
) -> UncertaintyReport:
    """Find where in LensCard space a trained model is weakest and propose the next batch.

    Joins the model's per-image scores on the given runs (evaluating first if needed) with the
    simulation records, bins each substructure class by mass fraction, SNR, axion mass (vortex)
    or subhalo count, and ranks the bins by mean uncertainty (classifier: 1 - P(true class);
    anomaly: how ordinary the reconstruction error looks). Returns the weakest cells and a
    suggested_card that adds images there; submit it with simulate_lens_batch (approval-gated),
    then retrain including the new run and evaluate again on the same held-out runs.

    Args:
        model_id: A trained model (see list_models).
        run_ids: Held-out runs to analyse (never the training runs).
        n_images_proposed: Size of the suggested follow-up batch.
        n_bins: Quantile bins per axis.
    """
    from lenscraft.ml.uncertainty import sample_uncertainty as _sample

    if model_id not in ctx.deps.backend.list_models():
        raise ModelRetry(f"unknown model_id {model_id!r}; see list_models")
    return _sample(ctx.deps.backend, model_id, run_ids, n_bins=n_bins, n_images_proposed=n_images_proposed)


# --------------------------------------------------------------------------------------------
# Toolsets
# --------------------------------------------------------------------------------------------


def full_toolset() -> FunctionToolset[AgentDeps]:
    """Simulation + ML tiers. Expensive tools are approval-gated."""
    ts: FunctionToolset[AgentDeps] = FunctionToolset()
    for fn in (simulate_lens_batch, train_classifier, train_anomaly_detector):
        ts.add_function(fn, requires_approval=True)
    for fn in (summarize_dataset, evaluate_model, sample_uncertainty, list_runs, list_models):
        ts.add_function(fn)
    return ts


def core_toolset() -> FunctionToolset[AgentDeps]:
    """Benchmark baseline: no domain tools (DESIGN.md Section 9, core-only arm)."""
    return FunctionToolset()
