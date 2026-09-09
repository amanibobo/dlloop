"""LensCraft: agentic gravitational lensing simulation pipeline.

Phase 1 surface area (see DESIGN.md, Section 11):

- :class:`lenscraft.schema.LensCard` -- the run-card the agent proposes and a human approves.
- :mod:`lenscraft.schema.lensjsonl` -- one-record-per-image LLM-legible dataset format.
- :func:`lenscraft.sim.simulate_lens_batch` -- executes an approved card against Lenstronomy.
"""

from lenscraft.schema import LensCard, LensRecord, MomentStats, SimBatchResult

__all__ = ["LensCard", "LensRecord", "MomentStats", "SimBatchResult"]
