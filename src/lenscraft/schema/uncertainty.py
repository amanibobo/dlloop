"""Schemas for the uncertainty-sampling loop (DESIGN.md Section 8)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from lenscraft.schema.lens_card import LensCard
from lenscraft.schema.ml import ModelKind


class UncertaintyCell(BaseModel):
    """One (substructure class, axis, bin) region of LensCard space and how the model does there."""

    model_config = ConfigDict(extra="forbid")

    substructure: str
    axis: str = Field(description="Binning axis: mass_fraction | snr | axion_mass | n_subhalos")
    low: float
    high: float
    n: int
    mean_uncertainty: float = Field(ge=0, le=1)
    score: float = Field(description="Classifier: accuracy in this cell. Anomaly: detection rate (none: 1 - false-alarm rate).")


class UncertaintyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    model_id: str
    kind: ModelKind = "classifier"
    run_ids: list[str] = Field(default_factory=list)
    n_images: int = 0
    overall_uncertainty: float = 0.0
    per_class_uncertainty: dict[str, float] = Field(default_factory=dict)
    cells: list[UncertaintyCell] = Field(default_factory=list, description="Weakest cells first (min n applied).")
    weakest: Optional[UncertaintyCell] = None
    suggested_card: Optional[LensCard] = Field(default=None, description="Batch that would add data where the model is weakest.")
    rationale: str = ""
    message: str = ""
