"""Run-time dependencies injected into every agent tool call."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lenscraft.compute.backend import ComputeBackend


@dataclass
class AgentDeps:
    backend: ComputeBackend
    wait_timeout: float | None = None  # seconds to wait for a remote job; None = until done
    poll_seconds: float = 2.0
    run_ids: list[str] = field(default_factory=list)  # runs created during this session
    model_ids: list[str] = field(default_factory=list)  # models trained during this session
    workspace: Path | None = None  # sandbox for the core-only arm's shell/file tools
