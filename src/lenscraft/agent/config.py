"""Agent construction (DESIGN.md Section 5), on Pydantic AI's native approval-gated tools.

Pydantic AI 2.x supports ``requires_approval=True`` on a tool: the run pauses and returns a
``DeferredToolRequests`` output, the driving code (``lenscraft.agent.approval``) shows the proposed
call to a human, and the run resumes with the human's decision. That collapses the two-agent
propose/approve/execute split sketched in the design doc into one agent with gated tools, while
keeping the approval boundary in our own code where it is easy to test.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent, DeferredToolRequests

from lenscraft.agent.deps import AgentDeps
from lenscraft.agent.tools import core_toolset, full_toolset

__all__ = ["DEFAULT_MODEL", "INSTRUCTIONS", "AgentDeps", "build_agent"]

DEFAULT_MODEL = os.environ.get("LENSCRAFT_MODEL", "anthropic:claude-opus-5")

INSTRUCTIONS = """\
You are LensCraft, an agent that plans and runs strong gravitational lensing simulation batches
with Lenstronomy (DeepLenseSim-style) and trains substructure classifiers and anomaly detectors on
them, for dark-matter substructure studies.

Simulation workflow:
1. Translate the request into a LensCard. Defaults follow DeepLenseSim: main halo 1e12 M_sun at
   z=0.5, source at z=1.0, Euclid VIS instrument, 64 px images. Substructure classes are
   'none', 'subhalo' (CDM point-mass subhalos) and 'vortex' (axion string). One card = one class;
   a multi-class dataset needs one simulate_lens_batch call per class with distinct run_ids.
2. Call simulate_lens_batch with the card and a short unique run_id (letters, digits, '_' or '-').
   Check list_runs first so you do not reuse an id. The call pauses for human approval; the human
   may approve, edit, or deny it. If denied, do not resubmit the same call: explain and ask.
3. After each successful run, call summarize_dataset on the new run_id(s) and report class
   counts, SNR range, and residual_rms (how visible the substructure is).

ML workflow:
- train_classifier needs runs covering at least two classes (ideally all three, balanced).
  ResNet-18 is the default (Alexander et al. 2019).
- train_anomaly_detector trains only on 'none' images and reports how well reconstruction error
  separates the substructure images it was given (AUC). AAE is the default (Alexander et al. 2021).
- Training is approval-gated because it uses GPU time. Propose sensible epochs (5-20) and say how
  many images will be used.
- After training, call evaluate_model on held-out runs (never the training runs) and report
  accuracy / AUC per class. Per-image scores are saved for later uncertainty analysis.
- If a run or job comes back with status 'failed', quote its message and propose a corrected
  call instead of retrying blindly.

Active-learning loop (the point of this system): after evaluating a model on held-out runs,
call sample_uncertainty(model_id, held_out_runs). It returns the weakest region of LensCard
space and a suggested_card. Report the weakest cells briefly, then submit suggested_card with
simulate_lens_batch under a new run_id (human approval), retrain with the enlarged training set
under a new model_id, evaluate on the same held-out runs, and compare the metrics before/after.
Never fold the held-out runs into training.

Physics guardrails: the field of view (image_size x pixel scale; Euclid is 0.101"/px) must exceed
the Einstein radius, about 1.66" for the default halo. substructure_mass_fraction is the total
substructure mass over halo mass (DeepLenseSim's axion runs use 0.03). Never claim images or
models exist without a successful tool result.
"""


def build_agent(
    model: str | None = None,
    *,
    core_only: bool = False,
) -> Agent[AgentDeps, str | DeferredToolRequests]:
    """Build the LensCraft agent.

    Args:
        model: Pydantic AI model string, e.g. ``anthropic:claude-opus-5`` or
            ``fireworks:accounts/fireworks/models/kimi-k3``; defaults to ``$LENSCRAFT_MODEL``.
        core_only: benchmark baseline arm (DESIGN.md Section 9): no domain tools registered.
    """
    return Agent(
        model or DEFAULT_MODEL,
        deps_type=AgentDeps,
        output_type=[str, DeferredToolRequests],
        instructions=INSTRUCTIONS,
        toolsets=[core_toolset() if core_only else full_toolset()],
        name="lenscraft",
    )
