"""Agent + approval loop, driven by a scripted FunctionModel (no API key, no network)."""

import pytest
from pydantic_ai import ToolApproved, ToolDenied
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lenscraft.agent import AgentDeps, auto_approver, build_agent, console_approver, run_with_approval
from lenscraft.agent.approval import _parse_patch
from lenscraft.compute import LocalBackend
from lenscraft.schema import LensCard

lenstronomy = pytest.importorskip("lenstronomy")
pytestmark = pytest.mark.integration

CARD = {"n_images": 1, "substructure": "vortex", "image_size": 32, "seed": 3}


def scripted_model(run_id="v1"):
    """Turn 1: propose a batch. Turn 2 (after the tool result): summarize. Turn 3: answer."""

    def fn(messages, info):
        last_parts = messages[-1].parts
        tool_returns = [p for p in last_parts if p.part_kind == "tool-return"]
        if not tool_returns:
            return ModelResponse(parts=[ToolCallPart("simulate_lens_batch", {"card": CARD, "run_id": run_id}, tool_call_id="c1")])
        if tool_returns[0].tool_name == "simulate_lens_batch":
            content = tool_returns[0].content
            if hasattr(content, "model_dump"):
                content = content.model_dump()
            if isinstance(content, dict) and content.get("status") == "ok":
                return ModelResponse(parts=[ToolCallPart("summarize_dataset", {"run_ids": [content["run_id"]]}, tool_call_id="c2")])
            return ModelResponse(parts=[TextPart(f"not run: {content}")])
        return ModelResponse(parts=[TextPart(f"summary: {tool_returns[0].content}")])

    return FunctionModel(fn)


def test_approved_batch_runs_and_is_summarized(tmp_path):
    agent = build_agent(scripted_model())
    deps = AgentDeps(backend=LocalBackend(tmp_path))
    result = run_with_approval(agent, "make one vortex image", deps, auto_approver)
    assert result.output.startswith("summary:") and "'vortex'" in result.output
    assert deps.run_ids == ["v1"]
    assert (tmp_path / "v1" / "images" / "00000.npy").exists()


def test_denied_batch_does_not_run(tmp_path):
    agent = build_agent(scripted_model())
    deps = AgentDeps(backend=LocalBackend(tmp_path))
    seen = []

    def deny(card, run_id):
        seen.append((card, run_id))
        return ToolDenied("human denied: too expensive")

    result = run_with_approval(agent, "make one vortex image", deps, deny)
    assert seen == [(LensCard.model_validate(CARD), "v1")]
    assert "too expensive" in result.output
    assert not (tmp_path / "v1").exists() and deps.run_ids == []


def test_human_edit_overrides_card(tmp_path):
    agent = build_agent(scripted_model())
    deps = AgentDeps(backend=LocalBackend(tmp_path))

    def edit(card, run_id):
        new = LensCard.model_validate({**card.model_dump(), "n_images": 2, "substructure": "subhalo"})
        return ToolApproved(override_args={"card": new.model_dump(), "run_id": "edited"})

    result = run_with_approval(agent, "make one vortex image", deps, edit)
    assert deps.run_ids == ["edited"]
    assert "'subhalo'" in result.output and "'n': 2" in result.output


def test_duplicate_run_id_is_retried_by_model(tmp_path):
    backend = LocalBackend(tmp_path)
    backend.submit(LensCard(n_images=1, substructure="none", image_size=32), "v1")
    calls = []

    def fn(messages, info):
        last = messages[-1].parts
        if any(p.part_kind == "retry-prompt" for p in last):
            calls.append("retry")
            return ModelResponse(parts=[ToolCallPart("simulate_lens_batch", {"card": CARD, "run_id": "v2"}, tool_call_id="c2")])
        if any(p.part_kind == "tool-return" for p in last):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("simulate_lens_batch", {"card": CARD, "run_id": "v1"}, tool_call_id="c1")])

    deps = AgentDeps(backend=backend)
    result = run_with_approval(build_agent(FunctionModel(fn)), "go", deps, auto_approver)
    assert result.output == "done" and calls == ["retry"] and deps.run_ids == ["v2"]


def test_core_only_agent_has_no_domain_tools():
    def fn(messages, info):
        return ModelResponse(parts=[TextPart(",".join(sorted(t.name for t in info.function_tools)) or "no-tools")])

    result = build_agent(FunctionModel(fn), core_only=True).run_sync("hi", deps=AgentDeps(backend=LocalBackend()))
    assert result.output == "no-tools"

    result = build_agent(FunctionModel(fn)).run_sync("hi", deps=AgentDeps(backend=LocalBackend()))
    assert result.output == "list_runs,simulate_lens_batch,summarize_dataset"


def test_console_approver_paths():
    card = LensCard(n_images=5, substructure="none")
    answers = iter(["y"])
    assert isinstance(console_approver(card, "r", input_fn=lambda _: next(answers)), ToolApproved)

    answers = iter(["n", "nope"])
    d = console_approver(card, "r", input_fn=lambda _: next(answers))
    assert isinstance(d, ToolDenied) and "nope" in d.message

    answers = iter(["e", "n_images=7 substructure=vortex run_id=r2", "y"])
    d = console_approver(card, "r", input_fn=lambda _: next(answers))
    assert isinstance(d, ToolApproved)
    assert d.override_args["card"]["n_images"] == 7 and d.override_args["card"]["substructure"] == "vortex"
    assert d.override_args["run_id"] == "r2"

    answers = iter(["e", "n_images=0", "y"])  # invalid edit -> re-prompt -> approve original
    assert isinstance(console_approver(card, "r", input_fn=lambda _: next(answers)), ToolApproved)


def test_parse_patch():
    assert _parse_patch('{"n_images": 3}') == {"n_images": 3}
    assert _parse_patch("seed=1 substructure=vortex halo_mass=1e11") == {"seed": 1, "substructure": "vortex", "halo_mass": 1e11}
    with pytest.raises(ValueError):
        _parse_patch("garbage")
