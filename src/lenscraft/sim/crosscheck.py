"""Cross-backend validation: render identical lens systems with lenstronomy and PyAutoLens and
measure how well the unconvolved model images agree (DESIGN.md Phase 5)."""

from __future__ import annotations

import numpy as np

from lenscraft.schema.crosscheck import CrossCheckImage, CrossCheckResult
from lenscraft.schema.lens_card import LensCard
from lenscraft.sim.batch import image_rng
from lenscraft.sim.lenstronomy_backend import build_lens_system


def lenstronomy_unconvolved(card: LensCard, system) -> np.ndarray:
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Util import util

    n, ps = card.image_size, card.pixel_scale
    _, _, ra0, dec0, _, _, pix2coord, _ = util.make_grid_with_coordtransform(num_pix=n, delta_pix=ps, center_ra=0, center_dec=0, subgrid_res=1, inverse=False)
    data = ImageData(image_data=np.zeros((n, n)), ra_at_xy_0=ra0, dec_at_xy_0=dec0, transform_pix2angle=pix2coord, background_rms=1.0, exposure_time=1.0)
    im = ImageModel(data, PSF(psf_type="NONE"), lens_model_class=LensModel(system.lens_model_list), source_model_class=LightModel(system.source_model_list),
                    kwargs_numerics={"supersampling_factor": 1})
    return np.asarray(im.image(system.kwargs_lens, system.kwargs_source, unconvolved=True), dtype=float)


def _centroid(a: np.ndarray) -> tuple[float, float]:
    a = np.clip(a, 0, None)
    y, x = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    s = a.sum()
    return (float((a * y).sum() / s), float((a * x).sum() / s)) if s > 0 else (0.0, 0.0)


def _centroid_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Flux-weighted centroid distance in pixels (the brightest pixel is ambiguous on a ring)."""
    (ya, xa), (yb, xb) = _centroid(a), _centroid(b)
    return float(np.hypot(ya - yb, xa - xb))


def cross_backend_validate(card: LensCard, n_images: int = 5, *, shape_tolerance: float = 0.05, flux_tolerance: float = 0.10) -> CrossCheckResult:
    """Compare the two engines on ``n_images`` systems drawn from ``card`` (custom instrument only)."""
    try:
        from lenscraft.sim import pyautolens_backend as pal
    except ImportError as exc:
        return CrossCheckResult(status="failed", card=card, message=str(exc))
    if card.instrument != "custom":
        return CrossCheckResult(status="failed", card=card, message="cross-check supports instrument='custom' only (set instrument='custom', pixel_scale, image_size)")
    try:
        pal._al()
    except ImportError as exc:
        return CrossCheckResult(status="failed", card=card, message=str(exc))

    images: list[CrossCheckImage] = []
    try:
        for i in range(n_images):
            system = build_lens_system(card, image_rng(card, i))
            L = lenstronomy_unconvolved(card, system)
            P = pal.model_image(card, system)
            ln, pn = L / L.sum(), P / P.sum()
            rms = float(np.sqrt(np.mean((pn - ln) ** 2)) / ln.max())
            images.append(CrossCheckImage(index=i, shape_rel_rms=rms, flux_ratio=float(P.sum() / L.sum()), centroid_shift_pixels=_centroid_shift(L, P)))
    except Exception as exc:  # noqa: BLE001
        return CrossCheckResult(status="failed", card=card, n_images=len(images), images=images, message=f"{type(exc).__name__}: {exc}")

    shapes = [im.shape_rel_rms for im in images]
    fluxes = [im.flux_ratio for im in images]
    agree = max(shapes) <= shape_tolerance and all(abs(f - 1.0) <= flux_tolerance for f in fluxes)
    return CrossCheckResult(
        card=card, n_images=len(images), images=images, mean_shape_rel_rms=float(np.mean(shapes)), max_shape_rel_rms=float(max(shapes)),
        mean_flux_ratio=float(np.mean(fluxes)), agree=agree, shape_tolerance=shape_tolerance, flux_tolerance=flux_tolerance, conventions=pal.CONVENTIONS,
    )
