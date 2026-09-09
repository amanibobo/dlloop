"""The human checkpoint: drive an agent run, pausing whenever it proposes an expensive step.

``run_with_approval`` is framework-thin and fully testable: give it any callable that turns a
proposed tool call ``(tool_name, args)`` into ``ToolApproved`` (optionally with edited args) or
``ToolDenied``. Gated tools are simulation batches and training jobs (see ``tools.APPROVAL_GATED``).
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

from lenscraft.agent.deps import AgentDeps
from lenscraft.schema.lens_card import LensCard

Decision = ToolApproved | ToolDenied
Approver = Callable[[str, dict[str, Any]], Decision]


def _validate_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalise proposed args so the human sees a validated object (raises ValidationError)."""
    if tool_name == "simulate_lens_batch":
        return {**args, "card": LensCard.model_validate(args.get("card", {})).model_dump()}
    return args


def run_with_approval(
    agent: Agent[AgentDeps, str | DeferredToolRequests],
    prompt: str,
    deps: AgentDeps,
    approve: Approver,
    *,
    message_history: Sequence[ModelMessage] | None = None,
    max_rounds: int = 20,
) -> AgentRunResult[str | DeferredToolRequests]:
    """Run until the agent produces text, resolving every approval request through ``approve``."""
    result = agent.run_sync(prompt, deps=deps, message_history=message_history)
    for _ in range(max_rounds):
        if not isinstance(result.output, DeferredToolRequests):
            return result
        approvals: dict[str, Decision] = {}
        for call in result.output.approvals:
            try:
                args = _validate_args(call.tool_name, call.args_as_dict())
            except ValidationError as exc:
                approvals[call.tool_call_id] = ToolDenied(f"proposed arguments failed validation: {exc}")
                continue
            approvals[call.tool_call_id] = approve(call.tool_name, args)
        result = agent.run_sync(
            deferred_tool_results=DeferredToolResults(approvals=approvals),
            message_history=result.all_messages(),
            deps=deps,
        )
    raise RuntimeError(f"agent still requesting approval after {max_rounds} rounds")


def auto_approver(tool_name: str, args: dict[str, Any]) -> Decision:
    """Approve everything (scripts, benchmarks)."""
    return ToolApproved()


def console_approver(
    tool_name: str,
    args: dict[str, Any],
    *,
    input_fn: Callable[[str], str] = input,
    out=sys.stderr,
) -> Decision:
    """Interactive terminal approval: y / n / e (edit fields as key=value pairs or a JSON patch).

    Edits apply to top-level arguments, or to fields of the nested LensCard for simulation batches.
    """
    while True:
        print(f"\n=== Proposed: {tool_name} ===", file=out)
        print(json.dumps(args, indent=2), file=out)
        answer = input_fn("Approve? [y]es / [n]o / [e]dit: ").strip().lower()
        if answer in ("y", "yes"):
            return ToolApproved()
        if answer in ("n", "no"):
            reason = input_fn("Reason (shown to the agent): ").strip() or "denied by human"
            return ToolDenied(f"human denied {tool_name}: {reason}")
        if answer in ("e", "edit"):
            raw = input_fn("Edits as key=value pairs (space-separated) or a JSON object: ").strip()
            try:
                edited = _apply_patch(tool_name, args, _parse_patch(raw))
            except (ValueError, ValidationError) as exc:
                print(f"invalid edit: {exc}", file=out)
                continue
            args = edited
            print("Edited. Review again:", file=out)
            if input_fn("Approve edited arguments? [y]/n: ").strip().lower() in ("", "y", "yes"):
                return ToolApproved(override_args=args)
            continue
        print("please answer y, n or e", file=out)


def _apply_patch(tool_name: str, args: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    new_args = dict(args)
    card = dict(args.get("card", {})) if "card" in args else None
    for k, v in patch.items():
        if k in args:
            new_args[k] = v
        elif card is not None:
            card[k] = v
        else:
            raise ValueError(f"unknown argument {k!r}; valid keys: {sorted(args)}")
    if card is not None:
        new_args["card"] = card
    return _validate_args(tool_name, new_args)


def _parse_patch(raw: str) -> dict[str, Any]:
    if raw.startswith("{"):
        return json.loads(raw)
    patch: dict[str, Any] = {}
    for token in raw.split():
        if "=" not in token:
            raise ValueError(f"expected key=value, got {token!r}")
        k, v = token.split("=", 1)
        try:
            patch[k] = json.loads(v)  # numbers, true/false/null, quoted strings, lists
        except json.JSONDecodeError:
            patch[k] = v
    return patch
