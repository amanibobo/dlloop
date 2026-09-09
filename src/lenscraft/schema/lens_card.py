"""LensCard: the structured simulation-batch config (DESIGN.md Section 4.1).

This is the analogue of a HEPTAPOD run card. The planning agent proposes one, a human
approves or edits it, and only then does it reach the execution tools.

Compared with the sketch in DESIGN.md, a few engine-facing fields were added so a card fully
determines a DeepLenseSim-style run:

- ``instrument`` / ``image_size`` / ``pixel_scale`` / ``psf_fwhm`` -- DeepLenseSim ships two
  observing setups: a Euclid VIS configuration from lenstronomy (Model II-IV, 64 px) and a
  hand-rolled 150 px / 0.05" setup (Model I, and the 150x150 inputs the 2021 autoencoders expect).
- ``source_magnitude`` -- DeepLenseSim controls depth through source magnitude and a randomly
  drawn exposure time rather than an SNR target; the *measured* SNR is recorded per image in the
  lensjsonl record instead, which is what the agent actually reasons over.
- ``axion_mass`` -- the vortex (axion) class needs an axion mass to set the vortex length.
- ``n_subhalos`` / ``subhalo_mass_min`` / ``subhalo_mass_max`` / ``subhalo_mass_slope`` -- the
  CDM subhalo mass function DeepLenseSim draws from.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SubstructureType = Literal["none", "subhalo", "vortex"]
BackendName = Literal["lenstronomy", "pyautolens"]
InstrumentName = Literal["euclid", "custom"]


class LensCard(BaseModel):
    """Configuration for one batch of strong-lensing simulations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- what to simulate -------------------------------------------------------------------
    n_images: int = Field(gt=0, le=100_000, description="Number of images in the batch.")
    substructure: SubstructureType = Field(
        description="Dark-matter substructure class: none, CDM point-mass subhalos, or axion vortex."
    )
    halo_mass: float = Field(
        default=1e12, gt=0, description="Main lens halo mass in solar masses (DeepLenseSim uses 1e12)."
    )
    redshift_lens: float = Field(default=0.5, gt=0)
    redshift_source: float = Field(default=1.0, gt=0)
    substructure_mass_fraction: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description=(
            "Total substructure mass as a fraction of halo_mass. Ignored for substructure='none'. "
            "DeepLenseSim's axion runs use 3e10 / 1e12 = 0.03."
        ),
    )

    # --- substructure details -----------------------------------------------------------------
    axion_mass: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Axion mass in eV (vortex only). None draws log-uniformly in [1e-24, 1e-22] per image, "
            "as DeepLenseSim does."
        ),
    )
    n_subhalos: int = Field(
        default=25, ge=1, description="Mean number of CDM subhalos in the field (Poisson mean)."
    )
    subhalo_mass_min: float = Field(default=1e6, gt=0, description="Subhalo mass-function lower bound, M_sun.")
    subhalo_mass_max: float = Field(default=1e10, gt=0, description="Subhalo mass-function upper bound, M_sun.")
    subhalo_mass_slope: float = Field(default=-0.9, description="Subhalo mass-function power-law slope.")

    # --- observation ----------------------------------------------------------------------------
    instrument: InstrumentName = Field(
        default="euclid",
        description="'euclid' = lenstronomy's Euclid VIS config (DeepLenseSim Model II+); 'custom' = generic setup below.",
    )
    image_size: int = Field(default=64, ge=16, le=1024, description="Image side length in pixels.")
    pixel_scale: float = Field(default=0.05, gt=0, description="Pixel scale in arcsec (custom instrument only).")
    psf_fwhm: float = Field(default=0.087, gt=0, description="Gaussian PSF FWHM in arcsec (custom instrument only).")
    source_magnitude: float = Field(default=20.0, description="Source apparent magnitude (euclid instrument).")
    background_rms: float = Field(default=1e-2, gt=0, description="Background noise rms per pixel (custom instrument).")

    # --- execution --------------------------------------------------------------------------------
    backend: BackendName = "lenstronomy"
    seed: Optional[int] = Field(default=None, ge=0, description="Master RNG seed. None = non-reproducible.")

    @model_validator(mode="after")
    def _check_physics(self) -> "LensCard":
        if self.redshift_source <= self.redshift_lens:
            raise ValueError("redshift_source must be greater than redshift_lens")
        if self.subhalo_mass_max <= self.subhalo_mass_min:
            raise ValueError("subhalo_mass_max must be greater than subhalo_mass_min")
        if self.substructure != "none" and self.substructure_mass_fraction == 0.0:
            raise ValueError("substructure_mass_fraction must be > 0 when substructure is not 'none'")
        return self

    @property
    def substructure_mass(self) -> float:
        """Total substructure mass in M_sun implied by this card."""
        if self.substructure == "none":
            return 0.0
        return self.substructure_mass_fraction * self.halo_mass


class SimBatchResult(BaseModel):
    """What ``simulate_lens_batch`` returns to the agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "failed"]
    run_id: str
    lensjsonl_path: str
    image_dir: str
    n_images: int
    elapsed_seconds: float
    message: str = ""
