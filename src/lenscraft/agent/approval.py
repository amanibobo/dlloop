"""The human checkpoint: drive an agent run, pausing whenever it proposes a simulation batch.

``run_with_approval`` is framework-thin and fully testable: give it any callable that turns a
proposed (LensCard, run_id) into ``ToolApproved`` (optionally with edits) or ``ToolDenied``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage

from lenscraft.agent.config import AgentDeps
from lenscraft.schema.lens_card import LensCard

Decision = ToolApproved | ToolDenied
Approver = Callable[[LensCard, str], Decision]


def _proposal(args: dict[str, Any]) -> tuple[LensCard, str]:
    return LensCard.model_validate(args.get("card", {})), str(args.get("run_id", ""))


def run_with_approval(
    agent: Agent[AgentDeps, str | DeferredToolRequests],
    prompt: str,
    deps: AgentDeps,
    approve: Approver,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    max_rounds: int = 10,
) -> AgentRunResult[str | DeferredToolRequests]:
    """Run until the agent produces text, resolving every approval request through ``approve``."""
    result = agent.run_sync(prompt, deps=deps, message_history=message_history)
    for _ in range(max_rounds):
        if not isinstance(result.output, DeferredToolRequests):
            return result
        approvals: dict[str, Decision] = {}
        for call in result.output.approvals:
            args = call.args_as_dict()
            try:
                card, run_id = _proposal(args)
            except ValidationError as exc:
                approvals[call.tool_call_id] = ToolDenied(f"proposed card failed validation: {exc}")
                continue
            approvals[call.tool_call_id] = approve(card, run_id)
        result = agent.run_sync(
            deferred_tool_results=DeferredToolResults(approvals=approvals),
            message_history=result.all_messages(),
            deps=deps,
        )
    raise RuntimeError(f"agent still requesting approval after {max_rounds} rounds")


def auto_approver(card: LensCard, run_id: str) -> Decision:
    """Approve everything (scripts, benchmarks)."""
    return ToolApproved()


def console_approver(card: LensCard, run_id: str, *, input_fn: Callable[[str], str] = input, out=sys.stderr) -> Decision:
    """Interactive terminal approval: y / n / e (edit fields as key=value or a JSON patch)."""
    while True:
        print(f"\n=== Proposed simulation batch: run_id={run_id!r} ===", file=out)
        print(card.model_dump_json(indent=2), file=out)
        answer = input_fn("Approve? [y]es / [n]o / [e]dit: ").strip().lower()
        if answer in ("y", "yes"):
            return ToolApproved()
        if answer in ("n", "no"):
            reason = input_fn("Reason (shown to the agent): ").strip() or "denied by human"
            return ToolDenied(f"human denied this batch: {reason}")
        if answer in ("e", "edit"):
            raw = input_fn("Edits as key=value pairs (space-separated) or a JSON object: ").strip()
            try:
                patch = _parse_patch(raw)
                new_run = str(patch.pop("run_id", run_id))
                new_card = LensCard.model_validate({**card.model_dump(), **patch})
            except (ValueError, ValidationError) as exc:
                print(f"invalid edit: {exc}", file=out)
                continue
            card = new_card
            run_id = new_run
            print("Edited. Review again:", file=out)
            if input_fn("Approve edited card? [y]/n: ").strip().lower() in ("", "y", "yes"):
                return ToolApproved(override_args={"card": card.model_dump(), "run_id": run_id})
            continue
        print("please answer y, n or e", file=out)


def _parse_patch(raw: str) -> dict[str, Any]:
    if raw.startswith("{"):
        return json.loads(raw)
    patch: dict[str, Any] = {}
    for token in raw.split():
        if "=" not in token:
            raise ValueError(f"expected key=value, got {token!r}")
        k, v = token.split("=", 1)
        try:
            patch[k] = json.loads(v)  # numbers, true/false/null, quoted strings
        except json.JSONDecodeError:
            patch[k] = v
    return patch
