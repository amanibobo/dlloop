"""Lenstronomy backend: a port of DeepLenseSim's ``deeplense/lens.py`` driven by a LensCard.

What is ported (see https://github.com/mwt5345/DeepLenseSim):

- main halo: SIE (e1=0.1, e2=0) + external shear (gamma1=0.05) at the lens redshift, with
  theta_E from the point-mass Einstein radius of ``halo_mass``;
- source: single elliptical Sersic (R=0.25", n=1, e=(-0.1, 0.1)), centre uniform in +-0.35";
- 'subhalo' (CDM): Poisson(n_subhalos) point masses, masses drawn from a power law, positions
  uniform in an annulus r in [0.25, 2.0]";
- 'vortex' (axion): the substructure mass split into 100 equal point masses on a line of
  half-length set by the axion mass, through the image centre at a random angle;
- 'euclid' instrument: lenstronomy's Euclid VIS configuration through ``SimAPI`` (Model II-IV);
- 'custom' instrument: hand-rolled ImageData/PSF/ImageModel at ``pixel_scale`` (Model I).

Deliberate deviations from DeepLenseSim (all bugs on their side):

1. Einstein radii use angular-diameter distances (see ``lenscraft.sim.cosmo``).
2. The CDM subhalo count actually follows the Poisson draw (DeepLenseSim draws it, then ignores it).
3. The Euclid image is model + noise, not model + model.
4. Subhalo masses are rescaled so their total equals ``substructure_mass_fraction * halo_mass``,
   so the card's mass fraction means the same thing for both substructure classes.
5. All randomness comes from a per-image ``numpy.random.Generator`` so a seeded card reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lenscraft.schema.lens_card import LensCard
from lenscraft.sim.cosmo import (
    DEFAULT_COSMOLOGY,
    axion_mass_to_vortex_length,
    point_mass_einstein_radius,
)

VORTEX_RESOLUTION = 100  # number of point masses along the vortex line (DeepLenseSim: res=100)


@dataclass
class LensSystem:
    """Everything lenstronomy needs for one image, plus the realised parameters we record."""

    lens_model_list: list[str]
    kwargs_lens: list[dict]
    n_smooth_components: int  # the first N entries are the smooth (macro) model
    source_model_list: list[str]
    kwargs_source: list[dict]
    realised: dict[str, float] = field(default_factory=dict)

    @property
    def smooth_lens_model_list(self) -> list[str]:
        return self.lens_model_list[: self.n_smooth_components]

    @property
    def smooth_kwargs_lens(self) -> list[dict]:
        return self.kwargs_lens[: self.n_smooth_components]


@dataclass
class RenderedImage:
    image: np.ndarray  # observed (noisy) image
    model: np.ndarray  # noiseless model with substructure
    smooth_model: np.ndarray  # noiseless model without substructure
    noise_sigma: np.ndarray  # per-pixel noise sigma estimate
    pixel_scale: float
    realised: dict[str, float]


# ----------------------------------------------------------------------------------------------
# System construction
# ----------------------------------------------------------------------------------------------


def build_lens_system(card: LensCard, rng: np.random.Generator) -> LensSystem:
    z_l, z_s = card.redshift_lens, card.redshift_source
    theta_e = point_mass_einstein_radius(card.halo_mass, z_l, z_s, DEFAULT_COSMOLOGY)

    lens_model_list = ["SIE", "SHEAR"]
    kwargs_lens = [
        {"theta_E": theta_e, "e1": 0.1, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
        {"gamma1": 0.05, "gamma2": 0.0},
    ]
    realised: dict[str, float] = {"theta_E": float(theta_e)}
    n_smooth = len(lens_model_list)

    if card.substructure == "subhalo":
        masses = _draw_subhalo_masses(card, rng)
        radii = point_mass_einstein_radius(masses, z_l, z_s, DEFAULT_COSMOLOGY)
        r = rng.uniform(0.25, 2.0, size=masses.size)
        th = rng.uniform(0.0, 2 * np.pi, size=masses.size)
        for te, x, y in zip(radii, r * np.sin(th), r * np.cos(th)):
            lens_model_list.append("POINT_MASS")
            kwargs_lens.append({"theta_E": float(te), "center_x": float(x), "center_y": float(y)})
        realised["n_subhalos"] = float(masses.size)
        realised["substructure_mass"] = float(masses.sum())

    elif card.substructure == "vortex":
        axion_mass = card.axion_mass if card.axion_mass is not None else 10 ** rng.uniform(-24, -22)
        half_length = axion_mass_to_vortex_length(axion_mass)
        element_mass = card.substructure_mass / VORTEX_RESOLUTION
        te = point_mass_einstein_radius(element_mass, z_l, z_s, DEFAULT_COSMOLOGY)
        coords = np.linspace(-half_length, half_length, VORTEX_RESOLUTION)
        ang = 2 * np.pi * rng.random()
        for s in coords:
            lens_model_list.append("POINT_MASS")
            kwargs_lens.append(
                {"theta_E": float(te), "center_x": float(s * np.cos(ang)), "center_y": float(s * np.sin(ang))}
            )
        realised["axion_mass"] = float(axion_mass)
        realised["vortex_half_length"] = float(half_length)
        realised["vortex_angle"] = float(ang)
        realised["substructure_mass"] = float(card.substructure_mass)

    else:
        realised["substructure_mass"] = 0.0

    cx, cy = rng.uniform(-0.35, 0.35, size=2)
    kwargs_source = {
        "R_sersic": 0.25,
        "n_sersic": 1.0,
        "e1": -0.1,
        "e2": 0.1,
        "center_x": float(cx),
        "center_y": float(cy),
    }
    if card.instrument == "euclid":
        kwargs_source["magnitude"] = card.source_magnitude
    else:
        kwargs_source["amp"] = 20.0  # DeepLenseSim simple_sim
    realised["source_center_x"] = float(cx)
    realised["source_center_y"] = float(cy)

    return LensSystem(
        lens_model_list=lens_model_list,
        kwargs_lens=kwargs_lens,
        n_smooth_components=n_smooth,
        source_model_list=["SERSIC_ELLIPSE"],
        kwargs_source=[kwargs_source],
        realised=realised,
    )


def _draw_subhalo_masses(card: LensCard, rng: np.random.Generator) -> np.ndarray:
    """Power-law subhalo masses, rescaled so the total equals the card's substructure mass."""
    n = max(1, int(rng.poisson(card.n_subhalos)))
    beta = card.subhalo_mass_slope
    u = rng.uniform(0.0, 1.0, size=n)
    lo, hi = card.subhalo_mass_min ** (beta + 1), card.subhalo_mass_max ** (beta + 1)
    masses = (lo + (hi - lo) * u) ** (1.0 / (beta + 1.0))
    return masses * (card.substructure_mass / masses.sum())


# ----------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------


EUCLID_PIXEL_SCALE = 0.101  # arcsec, fixed by lenstronomy's Euclid VIS configuration


def effective_pixel_scale(card: LensCard) -> float:
    return EUCLID_PIXEL_SCALE if card.instrument == "euclid" else card.pixel_scale


def check_field_of_view(card: LensCard, theta_e: float) -> float:
    """Return the field of view in arcsec, or raise if the arcs cannot land inside the image.

    The lensed arcs sit at radius ~theta_E from the centre, so a field narrower than theta_E
    contains none of them and silently produces a dataset of pure noise, which is exactly the
    kind of failure an agent should see up front. (The full ring fits only when fov >= 2 theta_E;
    partial arcs are still perfectly usable, so that is not enforced.)
    """
    fov = card.image_size * effective_pixel_scale(card)
    if fov < theta_e:
        raise ValueError(
            f"field of view {fov:.2f}\" ({card.image_size} px x {effective_pixel_scale(card):.3f}\"/px) is "
            f"smaller than the Einstein radius {theta_e:.2f}\", so the arcs fall outside the image; "
            "increase image_size or pixel_scale"
        )
    return fov


def render(card: LensCard, system: LensSystem, rng: np.random.Generator) -> RenderedImage:
    system.realised["field_of_view"] = check_field_of_view(card, system.realised["theta_E"])
    if card.instrument == "euclid":
        return _render_euclid(card, system, rng)
    return _render_custom(card, system, rng)


def _render_euclid(card: LensCard, system: LensSystem, rng: np.random.Generator) -> RenderedImage:
    from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
    from lenstronomy.SimulationAPI.sim_api import SimAPI

    kwargs_single_band = Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()
    kwargs_numerics = {"point_source_supersampling_factor": 1}

    def _image(lens_list: list[str], kwargs_lens: list[dict]) -> tuple[SimAPI, np.ndarray]:
        kwargs_model = {"lens_model_list": lens_list, "source_light_model_list": system.source_model_list}
        sim = SimAPI(num_pix=card.image_size, kwargs_single_band=kwargs_single_band, kwargs_model=kwargs_model)
        im_sim = sim.image_model_class(kwargs_numerics)
        _, kwargs_source, _ = sim.magnitude2amplitude(None, system.kwargs_source, None)
        return sim, im_sim.image(kwargs_lens, kwargs_source, None, None)

    sim, model = _image(system.lens_model_list, system.kwargs_lens)
    _, smooth = _image(system.smooth_lens_model_list, system.smooth_kwargs_lens)

    noise = sim.noise_for_model(model=model, seed=int(rng.integers(0, 2**31 - 1)))
    sigma = np.asarray(sim.estimate_noise(model), dtype=float)
    realised = dict(system.realised)
    realised["exposure_time"] = float(kwargs_single_band["exposure_time"])
    return RenderedImage(
        image=(model + noise).astype(np.float32),
        model=np.asarray(model, dtype=np.float32),
        smooth_model=np.asarray(smooth, dtype=np.float32),
        noise_sigma=sigma,
        pixel_scale=float(kwargs_single_band["pixel_scale"]),  # == EUCLID_PIXEL_SCALE
        realised=realised,
    )


def _render_custom(card: LensCard, system: LensSystem, rng: np.random.Generator) -> RenderedImage:
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util

    num_pix, delta_pix = card.image_size, card.pixel_scale
    exp_time = float(10 ** rng.uniform(3.0, 3.5))  # DeepLenseSim simple_sim
    background_rms = card.background_rms

    _, _, ra0, dec0, _, _, pix2coord, _ = util.make_grid_with_coordtransform(
        num_pix=num_pix, delta_pix=delta_pix, center_ra=0, center_dec=0, subgrid_res=1, inverse=False
    )
    data_class = ImageData(
        background_rms=background_rms,
        exposure_time=exp_time,
        ra_at_xy_0=ra0,
        dec_at_xy_0=dec0,
        transform_pix2angle=pix2coord,
        image_data=np.zeros((num_pix, num_pix)),
    )
    psf_class = PSF(psf_type="GAUSSIAN", fwhm=card.psf_fwhm, pixel_size=delta_pix, truncation=3)
    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}
    source_class = LightModel(system.source_model_list)

    def _image(lens_list: list[str], kwargs_lens: list[dict]) -> np.ndarray:
        im = ImageModel(
            data_class,
            psf_class,
            lens_model_class=LensModel(lens_list),
            source_model_class=source_class,
            lens_light_model_class=None,
            kwargs_numerics=kwargs_numerics,
        )
        return im.image(kwargs_lens, system.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)

    model = _image(system.lens_model_list, system.kwargs_lens)
    smooth = _image(system.smooth_lens_model_list, system.smooth_kwargs_lens)

    # model is in counts/s; realise Poisson noise on counts and Gaussian background
    counts = rng.poisson(np.clip(model, 0, None) * exp_time) / exp_time
    image = counts + rng.normal(0.0, background_rms, size=model.shape)
    sigma = np.sqrt(background_rms**2 + np.clip(model, 0, None) / exp_time)

    realised = dict(system.realised)
    realised["exposure_time"] = exp_time
    return RenderedImage(
        image=image.astype(np.float32),
        model=np.asarray(model, dtype=np.float32),
        smooth_model=np.asarray(smooth, dtype=np.float32),
        noise_sigma=sigma,
        pixel_scale=delta_pix,
        realised=realised,
    )


def simulate_one(card: LensCard, rng: np.random.Generator) -> RenderedImage:
    """Build and render a single lens system for this card using the given RNG."""
    system = build_lens_system(card, rng)
    return render(card, system, rng)
