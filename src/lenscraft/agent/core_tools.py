"""Core-only toolset for the benchmark baseline arm (DESIGN.md Section 9).

The baseline agent gets the same model, prompt and sandbox as the tool-assisted agent, but only
a shell and file tools: it has to drive lenstronomy and torch itself. Using the ``lenscraft``
package is blocked so the comparison measures the value of the domain tools, not of importing them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from lenscraft.agent.deps import AgentDeps

BLOCKED = ("lenscraft",)
MAX_OUTPUT = 8000


def _workspace(ctx: RunContext[AgentDeps]) -> Path:
    if ctx.deps.workspace is None:
        raise ModelRetry("no workspace configured for shell/file tools")
    return Path(ctx.deps.workspace).resolve()  # resolved: macOS tmp dirs live behind a /private symlink


def _resolve(ctx: RunContext[AgentDeps], path: str) -> Path:
    ws = _workspace(ctx)
    p = (ws / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if ws not in p.parents and p != ws:
        raise ModelRetry(f"path {path!r} is outside the workspace {ws}")
    return p


def _check_blocked(text: str) -> None:
    for word in BLOCKED:
        if word in text:
            raise ModelRetry(f"use of {word!r} is not allowed in this environment; write the pipeline yourself with lenstronomy/torch")


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[: MAX_OUTPUT // 2] + f"\n... [{len(s) - MAX_OUTPUT} chars omitted] ...\n" + s[-MAX_OUTPUT // 2 :]


def run_shell(ctx: RunContext[AgentDeps], command: str, timeout_seconds: int = 600) -> dict:
    """Run a shell command in the workspace and return its exit code, stdout and stderr.

    `python` on PATH is the project interpreter with numpy, scipy, astropy, lenstronomy, torch,
    torchvision and scikit-learn installed.

    Args:
        command: The command line to run (bash).
        timeout_seconds: Kill the command after this many seconds.
    """
    _check_blocked(command)
    ws = _workspace(ctx)
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/local/bin", "HOME": str(ws), "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.run(command, shell=True, cwd=ws, capture_output=True, text=True, timeout=timeout_seconds, env=env)
    except subprocess.TimeoutExpired as exc:
        return {"exit_code": -1, "stdout": _clip(exc.stdout or ""), "stderr": f"timed out after {timeout_seconds}s"}
    return {"exit_code": proc.returncode, "stdout": _clip(proc.stdout), "stderr": _clip(proc.stderr)}


def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Write a text file (e.g. a Python script) inside the workspace; parent directories are created.

    Args:
        path: Path relative to the workspace.
        content: Full file contents.
    """
    _check_blocked(content)
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p.relative_to(_workspace(ctx))}"


def read_file(ctx: RunContext[AgentDeps], path: str, max_chars: int = 8000) -> str:
    """Read a text file inside the workspace (truncated to max_chars).

    Args:
        path: Path relative to the workspace.
        max_chars: Maximum characters to return.
    """
    p = _resolve(ctx, path)
    if not p.exists():
        raise ModelRetry(f"{path!r} does not exist")
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= max_chars else text[:max_chars] + f"\n... [{len(text) - max_chars} chars omitted]"


def list_files(ctx: RunContext[AgentDeps], path: str = ".") -> list[str]:
    """List files under a workspace directory (recursive, at most 200 entries).

    Args:
        path: Directory relative to the workspace.
    """
    p = _resolve(ctx, path)
    if not p.exists():
        return []
    ws = _workspace(ctx)
    return sorted(str(q.relative_to(ws)) for q in p.rglob("*") if q.is_file())[:200]


def core_toolset() -> FunctionToolset[AgentDeps]:
    ts: FunctionToolset[AgentDeps] = FunctionToolset()
    for fn in (run_shell, write_file, read_file, list_files):
        ts.add_function(fn)
    return ts
