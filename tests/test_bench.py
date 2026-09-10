"""Rubric, estimators, core tools, and the trial driver with scripted agents (no LLM)."""

import json

import numpy as np
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lenscraft.bench import BenchTask, grade_workspace, pass_at_k, pass_pow_k
from lenscraft.bench.run_benchmark import format_summary, run_benchmark, run_trial, summarize
from lenscraft.schema import LensCard

TASK = BenchTask(n_train_per_class=2, n_test_per_class=1, image_size=32, input_size=96)


def test_estimators():
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
    assert pass_at_k(10, 5, 3) == pytest.approx(1 - (5 * 4 * 3) / (10 * 9 * 8))
    assert pass_pow_k(10, 5, 3) == pytest.approx((5 * 4 * 3) / (10 * 9 * 8))
    assert pass_pow_k(10, 2, 3) == 0.0 and pass_at_k(10, 8, 3) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def _fake_run(data_dir, run_id, cls, n, bad_label=False, missing_image=False):
    d = data_dir / run_id / "images"
    d.mkdir(parents=True)
    lines = []
    for i in range(n):
        if not (missing_image and i == 0):
            np.save(d / f"{i}.npy", np.zeros((4, 4), dtype=np.float32))
        lines.append(json.dumps({"image_id": f"{run_id}_{i}", "substructure_type": "vortex" if bad_label else cls, "image_path": f"images/{i}.npy"}))
    (data_dir / run_id / "records.lensjsonl").write_text("\n".join(lines) + "\n")


def _fake_model(data_dir, run_ids, auc, eval_runs, status="ok"):
    m = data_dir / "_models" / TASK.model_id
    m.mkdir(parents=True, exist_ok=True)
    (m / "train_result.json").write_text(json.dumps({"status": status, "run_ids": run_ids}))
    (m / "eval_test.json").write_text(json.dumps({"auc": auc, "run_ids": eval_runs}))


def test_rubric_stages(tmp_path):
    assert grade_workspace(tmp_path, TASK).reach == 0.0
    for c in TASK.classes:
        _fake_run(tmp_path, TASK.train_run(c), c, 2)
        _fake_run(tmp_path, TASK.test_run(c), c, 1)
    r = grade_workspace(tmp_path, TASK)
    assert r.stages_passed == ["simulate", "label"] and r.reach == pytest.approx(0.40)

    train = [TASK.train_run(c) for c in TASK.classes]
    test = [TASK.test_run(c) for c in TASK.classes]
    _fake_model(tmp_path, train + [test[0]], 0.9, test)  # leaked a test run into training
    assert grade_workspace(tmp_path, TASK).stages_passed == ["simulate", "label"]
    _fake_model(tmp_path, train, 0.3, test)  # too low
    r = grade_workspace(tmp_path, TASK)
    assert r.stages_passed == ["simulate", "label", "train"] and r.auc == 0.3 and not r.passed
    _fake_model(tmp_path, train, 0.9, train)  # evaluated on training data: does not count
    assert grade_workspace(tmp_path, TASK).auc is None
    _fake_model(tmp_path, train, 0.9, test)
    r = grade_workspace(tmp_path, TASK)
    assert r.passed and r.reach == 1.0 and r.auc == 0.9


def test_rubric_rejects_bad_labels_and_missing_images(tmp_path):
    for c in TASK.classes:
        _fake_run(tmp_path, TASK.train_run(c), c, 2, bad_label=(c == "none"))
        _fake_run(tmp_path, TASK.test_run(c), c, 1)
    assert grade_workspace(tmp_path, TASK).stages_passed == ["simulate"]
    for c in TASK.classes:
        _fake_run(tmp_path / "b", TASK.train_run(c), c, 2, missing_image=(c == "none"))
        _fake_run(tmp_path / "b", TASK.test_run(c), c, 1)
    assert grade_workspace(tmp_path / "b", TASK).stages_passed == []


def test_core_tools_sandbox(tmp_path):
    from pydantic_ai import ModelRetry

    from lenscraft.agent.core_tools import list_files, read_file, run_shell, write_file
    from lenscraft.agent.deps import AgentDeps
    from lenscraft.compute import LocalBackend

    class Ctx:
        deps = AgentDeps(backend=LocalBackend(tmp_path / "data"), workspace=tmp_path)

    assert "wrote" in write_file(Ctx, "s.py", "print(6*7)")
    out = run_shell(Ctx, "python s.py")
    assert out["exit_code"] == 0 and out["stdout"].strip() == "42"
    assert read_file(Ctx, "s.py") == "print(6*7)" and list_files(Ctx) == ["s.py"]
    with pytest.raises(ModelRetry):
        write_file(Ctx, "../escape.py", "x")
    with pytest.raises(ModelRetry):
        run_shell(Ctx, "python -c 'import lenscraft'")
    assert run_shell(Ctx, "sleep 5", timeout_seconds=1)["exit_code"] == -1


def _scripted_tool_arm_agent(model, core_only=False):
    """A perfect tool-arm agent: simulates all runs, trains, evaluates, answers."""
    from lenscraft.agent.config import build_agent

    plan = []
    for c in TASK.classes:
        plan.append(("simulate_lens_batch", {"card": {"n_images": TASK.n_train_per_class, "substructure": c, "image_size": 32, "seed": 1}, "run_id": TASK.train_run(c)}))
        plan.append(("simulate_lens_batch", {"card": {"n_images": TASK.n_test_per_class, "substructure": c, "image_size": 32, "seed": 2}, "run_id": TASK.test_run(c)}))
    plan.append(("train_classifier", {"model_id": TASK.model_id, "run_ids": [TASK.train_run(c) for c in TASK.classes], "epochs": 1}))
    plan.append(("evaluate_model", {"model_id": TASK.model_id, "run_ids": [TASK.test_run(c) for c in TASK.classes]}))

    def fn(messages, info):
        n_calls = sum(1 for m in messages for p in m.parts if p.part_kind == "tool-call")
        if n_calls < len(plan):
            name, args = plan[n_calls]
            return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f"c{n_calls}")])
        return ModelResponse(parts=[TextPart("done: AUC reported")])

    return build_agent(FunctionModel(fn), core_only=core_only)


@pytest.mark.integration
def test_trial_and_benchmark_with_scripted_agent(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lenstronomy")
    res = run_trial("tool", 1, TASK, workspace=tmp_path / "ws", agent_factory=_scripted_tool_arm_agent)
    assert not res.error, res.error
    assert res.rubric.stages_passed[:3] == ["simulate", "label", "train"]
    assert res.approvals == 7 and res.requests >= 8

    def lazy_agent(model, core_only=False):
        return _scripted_tool_arm_agent(model, core_only)

    results, summary = run_benchmark(arms=("tool",), trials=2, task=TASK, out_dir=tmp_path / "out", agent_factory=lazy_agent, progress=lambda s: None)
    assert len(results) == 2 and summary[0].n == 2
    assert (tmp_path / "out" / "summary.json").exists() and (tmp_path / "out" / "results.jsonl").read_text().count("\n") == 2
    assert "pass@1" in format_summary(summary)


def test_core_arm_trial_is_graded_not_crashed(tmp_path):
    """A core-arm agent that just answers produces a graded zero, not an exception."""
    from lenscraft.agent.config import build_agent

    def fn(messages, info):
        names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("tools: " + ",".join(names))])

    res = run_trial("core", 1, TASK, workspace=tmp_path / "ws", agent_factory=lambda m, core_only=False: build_agent(FunctionModel(fn), core_only=core_only))
    assert res.final_text == "tools: list_files,read_file,run_shell,write_file"
    assert res.rubric.reach == 0.0 and not res.error
    assert summarize([res])[0].pass_at_k["1"] == 0.0
