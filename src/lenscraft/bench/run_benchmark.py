"""N-trial benchmark driver: tool-assisted vs core-only arms (DESIGN.md Section 9)."""

from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import UsageLimits

from lenscraft.agent.approval import Approver, auto_approver, run_with_approval
from lenscraft.agent.config import build_agent
from lenscraft.agent.deps import AgentDeps
from lenscraft.bench.rubric import STAGES, RubricResult, grade_workspace, pass_at_k, pass_pow_k
from lenscraft.bench.task import BenchTask
from lenscraft.compute.backend import LocalBackend

ARMS = ("tool", "core")


class TrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: str
    trial: int
    model: str
    rubric: RubricResult
    duration_seconds: float
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    approvals: int = 0
    error: str = ""
    final_text: str = ""
    workspace: str = ""


class ArmSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: str
    n: int
    successes: int
    pass_at_k: dict[str, float]
    pass_pow_k: dict[str, float]
    reach_mean: float
    reach_std: float
    stage_pass_rate: dict[str, float]
    mean_duration_seconds: float
    mean_requests: float
    mean_tokens: float
    errors: int


def run_trial(
    arm: str,
    trial: int,
    task: BenchTask,
    *,
    model: str | None = None,
    workspace: Path,
    approve: Approver = auto_approver,
    request_limit: int = 60,
    agent_factory=build_agent,
) -> TrialResult:
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / "data"
    backend = LocalBackend(data_dir)
    agent = agent_factory(model, core_only=(arm == "core"))
    deps = AgentDeps(backend=backend, workspace=workspace)
    approvals = 0

    def counting_approver(tool_name: str, args: dict[str, Any]):
        nonlocal approvals
        approvals += 1
        return approve(tool_name, args)

    t0 = time.perf_counter()
    error, final_text = "", ""
    requests = in_tok = out_tok = 0
    try:
        result = run_with_approval(
            agent, task.prompt(arm, workspace), deps, counting_approver, max_rounds=request_limit,
            usage_limits=UsageLimits(request_limit=request_limit),
        )
        final_text = str(result.output)
        # usage is per run and every approval resume is a new run: total it over the whole history
        for msg in result.all_messages():
            if getattr(msg, "kind", "") == "response":
                requests += 1
                u = getattr(msg, "usage", None)
                in_tok += getattr(u, "input_tokens", 0) or 0
                out_tok += getattr(u, "output_tokens", 0) or 0
    except Exception as exc:  # noqa: BLE001 - a crashed trial is a failed trial, not a crashed benchmark
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    duration = time.perf_counter() - t0
    rubric = grade_workspace(data_dir, task, final_text)
    model_name = getattr(getattr(agent, "model", None), "model_name", None) or str(model)
    return TrialResult(
        arm=arm, trial=trial, model=str(model_name), rubric=rubric, duration_seconds=duration,
        requests=requests, input_tokens=in_tok, output_tokens=out_tok, approvals=approvals, error=error,
        final_text=final_text[:2000], workspace=str(workspace),
    )


def summarize(results: list[TrialResult], ks: tuple[int, ...] = (1, 3, 5)) -> list[ArmSummary]:
    out = []
    for arm in sorted({r.arm for r in results}):
        rs = [r for r in results if r.arm == arm]
        n, c = len(rs), sum(r.rubric.passed for r in rs)
        reaches = [r.rubric.reach for r in rs]
        out.append(
            ArmSummary(
                arm=arm, n=n, successes=c,
                pass_at_k={str(k): pass_at_k(n, c, k) for k in ks if k <= n},
                pass_pow_k={str(k): pass_pow_k(n, c, k) for k in ks if k <= n},
                reach_mean=statistics.fmean(reaches), reach_std=statistics.pstdev(reaches) if n > 1 else 0.0,
                stage_pass_rate={s: sum(s in r.rubric.stages_passed for r in rs) / n for s, _ in STAGES},
                mean_duration_seconds=statistics.fmean(r.duration_seconds for r in rs),
                mean_requests=statistics.fmean(r.requests for r in rs),
                mean_tokens=statistics.fmean(r.input_tokens + r.output_tokens for r in rs),
                errors=sum(bool(r.error) for r in rs),
            )
        )
    return out


def run_benchmark(
    *,
    arms: tuple[str, ...] = ARMS,
    trials: int = 10,
    task: BenchTask | None = None,
    model: str | None = None,
    out_dir: str | Path = "reports/bench",
    keep_workspaces: bool = False,
    request_limit: int = 60,
    agent_factory=build_agent,
    progress=print,
) -> tuple[list[TrialResult], list[ArmSummary]]:
    task = task or BenchTask()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results: list[TrialResult] = []
    with results_path.open("a", encoding="utf-8") as fh:
        for trial in range(1, trials + 1):
            for arm in arms:
                ws = Path(tempfile.mkdtemp(prefix=f"lenscraft-bench-{arm}-{trial}-"))
                res = run_trial(arm, trial, task, model=model, workspace=ws, request_limit=request_limit, agent_factory=agent_factory)
                results.append(res)
                fh.write(res.model_dump_json() + "\n")
                fh.flush()
                progress(
                    f"[{arm} #{trial}] {'PASS' if res.rubric.passed else 'fail'} reach={res.rubric.reach:.2f} "
                    f"stages={res.rubric.stages_passed} auc={res.rubric.auc} {res.duration_seconds:.0f}s req={res.requests}"
                    + (f" ERROR {res.error.splitlines()[0]}" if res.error else "")
                )
                if not keep_workspaces:
                    shutil.rmtree(ws, ignore_errors=True)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps({"task": task.model_dump(), "arms": [s.model_dump() for s in summary]}, indent=2), encoding="utf-8")
    return results, summary


def format_summary(summary: list[ArmSummary]) -> str:
    lines = ["arm    n  pass  pass@1 pass@3 reach     simulate label train auc   req   tokens   time"]
    for s in summary:
        lines.append(
            f"{s.arm:5s} {s.n:3d} {s.successes:4d}  {s.pass_at_k.get('1', 0):.2f}   {s.pass_at_k.get('3', float('nan')):.2f}   "
            f"{s.reach_mean:.2f}±{s.reach_std:.2f} "
            f"{s.stage_pass_rate['simulate']:.2f}     {s.stage_pass_rate['label']:.2f}  {s.stage_pass_rate['train']:.2f}  {s.stage_pass_rate['auc']:.2f}  "
            f"{s.mean_requests:4.0f} {s.mean_tokens:8.0f} {s.mean_duration_seconds:5.0f}s"
        )
    return "\n".join(lines)
