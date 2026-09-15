"""Schemas for cross-backend validation (DESIGN.md Section 6, `cross_backend_validate`)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from lenscraft.schema.lens_card import LensCard


class CrossCheckImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    shape_rel_rms: float = Field(description="rms of (PyAutoLens - lenstronomy) on flux-normalised unconvolved models, over the lenstronomy peak.")
    flux_ratio: float = Field(description="PyAutoLens total / lenstronomy total, after converting surface brightness to counts per pixel.")
    centroid_shift_pixels: float = Field(description="Distance between the flux-weighted centroids of the two models, in pixels.")


class CrossCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    card: Optional[LensCard] = None
    n_images: int = 0
    images: list[CrossCheckImage] = Field(default_factory=list)
    mean_shape_rel_rms: float = 0.0
    max_shape_rel_rms: float = 0.0
    mean_flux_ratio: float = 0.0
    agree: bool = False
    shape_tolerance: float = 0.05
    flux_tolerance: float = 0.10
    conventions: str = ""
    message: str = ""
