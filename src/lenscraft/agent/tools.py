"""Domain tools (DESIGN.md Section 6). Phase 2 registers the simulation tier; ML and docs tiers
are added in Phases 3 and 6. Tools only talk to ``ComputeBackend``, never to the engine directly.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from lenscraft.agent.deps import AgentDeps
from lenscraft.schema.lens_card import LensCard, SimBatchResult
from lenscraft.schema.lensjsonl import summarize_records

RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,39}")


def simulate_lens_batch(ctx: RunContext[AgentDeps], card: LensCard, run_id: str) -> SimBatchResult:
    """Generate a batch of strong-lensing images from a LensCard. Requires human approval.

    Args:
        card: The simulation configuration (substructure class, halo mass, redshifts, instrument, seed, ...).
        run_id: Short unique name for this run, e.g. 'vortex_a'. Letters, digits, '_' or '-' only.
    """
    if not RUN_ID_RE.fullmatch(run_id):
        raise ModelRetry(f"run_id {run_id!r} is invalid: use 1-40 characters from letters, digits, '_' or '-'")
    if run_id in ctx.deps.backend.list_runs():
        raise ModelRetry(f"run_id {run_id!r} already exists; pick a new one (see list_runs)")

    job_id = ctx.deps.backend.submit(card, run_id)
    status = ctx.deps.backend.wait(job_id, poll_seconds=ctx.deps.poll_seconds, timeout=ctx.deps.wait_timeout)
    if status.result is None:
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


def full_toolset() -> FunctionToolset[AgentDeps]:
    """Simulation tier. ``simulate_lens_batch`` is approval-gated."""
    ts: FunctionToolset[AgentDeps] = FunctionToolset()
    ts.add_function(simulate_lens_batch, requires_approval=True)
    ts.add_function(summarize_dataset)
    ts.add_function(list_runs)
    return ts


def core_toolset() -> FunctionToolset[AgentDeps]:
    """Benchmark baseline: no domain tools (DESIGN.md Section 9, core-only arm)."""
    return FunctionToolset()
