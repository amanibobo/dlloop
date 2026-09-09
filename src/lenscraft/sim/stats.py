"""Cheap, generation-time image statistics for lensjsonl records (DESIGN.md Section 4.2)."""

from __future__ import annotations

import numpy as np

from lenscraft.schema.lensjsonl import MomentStats


def _moment_tensor(image: np.ndarray, pixel_scale: float) -> tuple[float, float, float]:
    """Flux-weighted second moments (Qxx, Qyy, Qxy) about the centroid, in arcsec^2."""
    img = np.clip(np.asarray(image, dtype=float), 0, None)
    total = img.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    x = (x - (nx - 1) / 2) * pixel_scale
    y = (y - (ny - 1) / 2) * pixel_scale
    cx = (img * x).sum() / total
    cy = (img * y).sum() / total
    dx, dy = x - cx, y - cy
    qxx = (img * dx * dx).sum() / total
    qyy = (img * dy * dy).sum() / total
    qxy = (img * dx * dy).sum() / total
    return float(qxx), float(qyy), float(qxy)


def compute_moment_stats(
    model_image: np.ndarray,
    smooth_model_image: np.ndarray,
    pixel_scale: float,
) -> MomentStats:
    """Build ``MomentStats`` from the noiseless full model and the noiseless smooth-lens model.

    Args:
        model_image: noiseless image with substructure included.
        smooth_model_image: noiseless image of the same system with substructure removed.
        pixel_scale: arcsec per pixel.
    """
    peak = float(np.max(smooth_model_image)) if smooth_model_image.size else 0.0
    if peak > 0:
        residual_rms = float(np.sqrt(np.mean((model_image - smooth_model_image) ** 2)) / peak)
    else:
        residual_rms = 0.0

    qxx, qyy, qxy = _moment_tensor(model_image, pixel_scale)
    trace = qxx + qyy
    second_moment = float(np.sqrt(trace)) if trace > 0 else 0.0
    if trace > 0:
        e1 = (qxx - qyy) / trace
        e2 = 2 * qxy / trace
        ellipticity = float(min(1.0, np.hypot(e1, e2)))
    else:
        ellipticity = 0.0

    return MomentStats(
        residual_rms=residual_rms,
        second_moment=second_moment,
        arc_ellipticity=ellipticity,
    )


def estimate_snr(model_image: np.ndarray, noise_sigma: np.ndarray | float, threshold_sigma: float = 3.0) -> float:
    """Integrated SNR of the lensed light.

    Pixels where the noiseless model exceeds ``threshold_sigma`` times the local noise are treated
    as signal; SNR = sum(signal) / sqrt(sum(noise variance)) over those pixels. Returns 0 if no
    pixel clears the threshold.
    """
    model = np.asarray(model_image, dtype=float)
    sigma = np.broadcast_to(np.asarray(noise_sigma, dtype=float), model.shape)
    mask = model > threshold_sigma * sigma
    if not mask.any():
        return 0.0
    var = float(np.sum(sigma[mask] ** 2))
    if var <= 0:
        return 0.0
    return float(np.sum(model[mask]) / np.sqrt(var))
