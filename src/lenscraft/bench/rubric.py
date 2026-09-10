"""Stagewise rubric and pass@k / pass^k estimators (DESIGN.md Section 9, after HEPTAPOD Sec. 5).

Grading looks only at what exists on disk in the trial's sandbox, so it applies equally to the
tool-assisted arm (which writes the lensjsonl layout natively) and the core arm (which is told
the layout in its prompt). Stages are cumulative: a stage only counts if every earlier one passed.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lenscraft.bench.task import BenchTask

# (name, weight): weights sum to 1; "reach" is the cumulative weight of passed stages.
STAGES: list[tuple[str, float]] = [
    ("simulate", 0.25),  # every requested run exists with enough readable images
    ("label", 0.15),  # records carry the right substructure label for their run
    ("train", 0.35),  # a model was trained, on training runs only
    ("auc", 0.25),  # evaluated on the held-out runs with AUC >= threshold
]


class RubricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages_passed: list[str] = Field(default_factory=list)
    reach: float = 0.0
    passed: bool = False
    auc: float | None = None
    notes: dict[str, str] = Field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _check_run(run_dir: Path, expected_cls: str, n_min: int) -> tuple[bool, bool, str]:
    """Returns (simulated_ok, labels_ok, note)."""
    rec_path = run_dir / "records.lensjsonl"
    if not rec_path.exists():
        return False, False, "no records.lensjsonl"
    try:
        recs = _read_jsonl(rec_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, False, f"unreadable records: {exc}"
    if len(recs) < n_min:
        return False, False, f"{len(recs)} records < {n_min}"
    n_img = 0
    for r in recs:
        p = run_dir / str(r.get("image_path", ""))
        if p.suffix == ".npy" and p.exists():
            try:
                arr = np.load(p)
                if arr.ndim == 2 and np.isfinite(arr).all():
                    n_img += 1
            except Exception:  # noqa: BLE001
                pass
    if n_img < n_min:
        return False, False, f"{n_img} valid images < {n_min}"
    labels = {str(r.get("substructure_type")) for r in recs}
    return True, labels == {expected_cls}, f"{len(recs)} records, labels {sorted(labels)}"


def grade_workspace(data_dir: str | Path, task: BenchTask) -> RubricResult:
    data_dir = Path(data_dir)
    result = RubricResult()
    notes = result.notes

    # --- simulate + label ----------------------------------------------------------------
    sim_ok, label_ok = True, True
    for cls in task.classes:
        for run_id, n_min in ((task.train_run(cls), task.n_train_per_class), (task.test_run(cls), task.n_test_per_class)):
            s, l, note = _check_run(data_dir / run_id, cls, n_min)
            notes[run_id] = note
            sim_ok &= s
            label_ok &= l
    if not sim_ok:
        return _finish(result, [])
    if not label_ok:
        return _finish(result, ["simulate"])

    # --- train --------------------------------------------------------------------------------
    model_dir = data_dir / "_models" / task.model_id
    tr_path = model_dir / "train_result.json"
    if not tr_path.exists():
        notes["train"] = "no train_result.json"
        return _finish(result, ["simulate", "label"])
    try:
        tr = json.loads(tr_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        notes["train"] = f"bad train_result.json: {exc}"
        return _finish(result, ["simulate", "label"])
    train_runs = set(map(str, tr.get("run_ids", [])))
    test_runs = {task.test_run(c) for c in task.classes}
    if tr.get("status") != "ok":
        notes["train"] = f"status {tr.get('status')!r}"
        return _finish(result, ["simulate", "label"])
    if not train_runs or train_runs & test_runs:
        notes["train"] = f"trained on {sorted(train_runs)} (must be training runs only)"
        return _finish(result, ["simulate", "label"])
    notes["train"] = f"ok on {sorted(train_runs)}"

    # --- auc -----------------------------------------------------------------------------------
    best: float | None = None
    for ev_path in sorted(model_dir.glob("eval_*.json")):
        try:
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ev_runs = set(map(str, ev.get("run_ids", [])))
        auc = ev.get("auc")
        if not isinstance(auc, (int, float)) or not ev_runs or not ev_runs <= test_runs or ev_runs & train_runs:
            continue
        best = auc if best is None else max(best, auc)
    result.auc = best
    if best is None:
        notes["auc"] = "no evaluation on the held-out runs"
        return _finish(result, ["simulate", "label", "train"])
    if best < task.auc_threshold:
        notes["auc"] = f"auc {best:.3f} < {task.auc_threshold}"
        return _finish(result, ["simulate", "label", "train"])
    notes["auc"] = f"auc {best:.3f}"
    return _finish(result, [s for s, _ in STAGES])


def _finish(result: RubricResult, passed: list[str]) -> RubricResult:
    weights = dict(STAGES)
    result.stages_passed = passed
    result.reach = float(sum(weights[s] for s in passed))
    result.passed = len(passed) == len(STAGES)
    return result


# --------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k sampled attempts succeeds), unbiased (Chen et al. 2021)."""
    if k > n:
        raise ValueError("k must be <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """P(all k sampled attempts succeed), unbiased."""
    if k > n:
        raise ValueError("k must be <= n")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)
