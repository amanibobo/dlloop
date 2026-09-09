"""Run-time dependencies injected into every agent tool call."""

from __future__ import annotations

from dataclasses import dataclass, field

from lenscraft.compute.backend import ComputeBackend


@dataclass
class AgentDeps:
    backend: ComputeBackend
    wait_timeout: float | None = None  # seconds to wait for a remote batch; None = until done
    poll_seconds: float = 2.0
    run_ids: list[str] = field(default_factory=list)  # runs created during this session
