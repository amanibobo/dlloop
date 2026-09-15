"""PyAutoLens backend (DESIGN.md Phase 5, stretch): renders the *same* ``LensSystem`` that the
lenstronomy backend builds, through PyAutoLens, so the two engines can be compared image by image.

Conventions, established empirically by minimising the residual between the two engines over
every orientation / sign combination (see ``lenscraft.sim.crosscheck``):

- ellipticity: PyAutoLens ``ell_comps = (e2, e1)`` for lenstronomy ``(e1, e2)``;
- external shear: ``(gamma_1, gamma_2) = (gamma1, gamma2)`` unchanged;
- centres: PyAutoLens ``(y, x)`` for lenstronomy ``(x, y)``;
- arrays: PyAutoLens' native image is flipped top-to-bottom relative to lenstronomy's;
- flux: PyAutoLens returns surface brightness; lenstronomy returns counts per pixel, i.e.
  surface brightness times the pixel area.

Only the ``custom`` instrument is supported (the Euclid path relies on lenstronomy's SimAPI
magnitude system). Requires ``autolens`` (``uv pip install autolens``); it is not a core dependency.
"""

from __future__ import annotations

import numpy as np

from lenscraft.schema.lens_card import LensCard
from lenscraft.sim.lenstronomy_backend import LensSystem, RenderedImage, build_lens_system, check_field_of_view

CONVENTIONS = "ell_comps=(e2,e1); shear unchanged; centre=(y,x); native array flipped vertically; flux x pixel_scale^2"


def _al():
    try:
        import autolens as al
    except ImportError as exc:  # pragma: no cover - exercised only where autolens is missing
        raise ImportError("the pyautolens backend needs the 'autolens' package (uv pip install autolens)") from exc
    return al


def to_pal_comps(e1: float, e2: float) -> tuple[float, float]:
    return (e2, e1)


def to_pal_centre(x: float, y: float) -> tuple[float, float]:
    return (y, x)


def from_pal_native(arr: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(arr))


def build_tracer(card: LensCard, system: LensSystem, smooth_only: bool = False):
    """PyAutoLens Tracer for a LensSystem (optionally without the substructure components)."""
    al = _al()
    lens_list = system.smooth_lens_model_list if smooth_only else system.lens_model_list
    kwargs_lens = system.smooth_kwargs_lens if smooth_only else system.kwargs_lens
    profiles: dict[str, object] = {}
    for i, (name, kw) in enumerate(zip(lens_list, kwargs_lens)):
        if name == "SIE":
            profiles[f"mass_{i}"] = al.mp.Isothermal(centre=to_pal_centre(kw["center_x"], kw["center_y"]), ell_comps=to_pal_comps(kw["e1"], kw["e2"]), einstein_radius=kw["theta_E"])
        elif name == "SHEAR":
            profiles[f"shear_{i}"] = al.mp.ExternalShear(gamma_1=kw["gamma1"], gamma_2=kw["gamma2"])
        elif name == "POINT_MASS":
            profiles[f"sub_{i}"] = al.mp.PointMass(centre=to_pal_centre(kw["center_x"], kw["center_y"]), einstein_radius=kw["theta_E"])
        else:
            raise ValueError(f"no PyAutoLens equivalent implemented for lenstronomy profile {name!r}")
    lens = al.Galaxy(redshift=card.redshift_lens, **profiles)

    ks = system.kwargs_source[0]
    if "amp" not in ks:
        raise ValueError("pyautolens backend supports the 'custom' instrument only (amplitude-based source)")
    source = al.Galaxy(
        redshift=card.redshift_source,
        light=al.lp.Sersic(centre=to_pal_centre(ks["center_x"], ks["center_y"]), ell_comps=to_pal_comps(ks["e1"], ks["e2"]),
                           intensity=ks["amp"], effective_radius=ks["R_sersic"], sersic_index=ks["n_sersic"]),
    )
    return al.Tracer(galaxies=[lens, source])


def model_image(card: LensCard, system: LensSystem, smooth_only: bool = False) -> np.ndarray:
    """Unconvolved noiseless model in lenstronomy's orientation and units (counts per pixel)."""
    al = _al()
    grid = al.Grid2D.uniform(shape_native=(card.image_size, card.image_size), pixel_scales=card.pixel_scale, over_sample_size=1)
    tracer = build_tracer(card, system, smooth_only)
    return from_pal_native(tracer.image_2d_from(grid).native) * card.pixel_scale**2


def render(card: LensCard, system: LensSystem, rng: np.random.Generator) -> RenderedImage:
    """PSF-convolved, noisy image with the same noise model as the lenstronomy 'custom' path."""
    from scipy.ndimage import gaussian_filter

    if card.instrument != "custom":
        raise ValueError("pyautolens backend supports instrument='custom' only")
    system.realised["field_of_view"] = check_field_of_view(card, system.realised["theta_E"])
    sigma_pix = card.psf_fwhm / 2.354820045 / card.pixel_scale
    model = gaussian_filter(model_image(card, system), sigma_pix, mode="constant")
    smooth = gaussian_filter(model_image(card, system, smooth_only=True), sigma_pix, mode="constant")

    exp_time = float(10 ** rng.uniform(3.0, 3.5))
    counts = rng.poisson(np.clip(model, 0, None) * exp_time) / exp_time
    image = counts + rng.normal(0.0, card.background_rms, size=model.shape)
    sigma = np.sqrt(card.background_rms**2 + np.clip(model, 0, None) / exp_time)
    realised = dict(system.realised)
    realised["exposure_time"] = exp_time
    return RenderedImage(
        image=image.astype(np.float32), model=model.astype(np.float32), smooth_model=smooth.astype(np.float32),
        noise_sigma=sigma, pixel_scale=card.pixel_scale, realised=realised,
    )


def simulate_one(card: LensCard, rng: np.random.Generator) -> RenderedImage:
    system = build_lens_system(card, rng)
    return render(card, system, rng)
