import numpy as np
import pytest

from lenscraft.sim.stats import compute_moment_stats, estimate_snr


def _gaussian(n=64, sx=4.0, sy=4.0, scale=0.1):
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    return np.exp(-0.5 * (((x - c) / sx) ** 2 + ((y - c) / sy) ** 2))


def test_round_blob_has_zero_ellipticity_and_expected_size():
    img = _gaussian(sx=4.0, sy=4.0)
    stats = compute_moment_stats(img, img, pixel_scale=0.1)
    assert stats.arc_ellipticity == pytest.approx(0.0, abs=1e-6)
    assert stats.residual_rms == 0.0
    # trace of second moments = sx^2 + sy^2 in pixels -> sqrt(32) px * 0.1"/px
    assert stats.second_moment == pytest.approx(np.sqrt(32) * 0.1, rel=1e-2)


def test_elongated_blob_is_elliptical():
    img = _gaussian(sx=8.0, sy=2.0)
    stats = compute_moment_stats(img, img, pixel_scale=0.1)
    # e1 = (64 - 4) / (64 + 4)
    assert stats.arc_ellipticity == pytest.approx(60 / 68, rel=1e-2)


def test_residual_rms_is_relative_to_smooth_peak():
    smooth = _gaussian()
    full = smooth + 0.05
    stats = compute_moment_stats(full, smooth, pixel_scale=0.1)
    assert stats.residual_rms == pytest.approx(0.05 / smooth.max(), rel=1e-6)


def test_empty_image_is_safe():
    z = np.zeros((16, 16))
    stats = compute_moment_stats(z, z, pixel_scale=0.1)
    assert (stats.residual_rms, stats.second_moment, stats.arc_ellipticity) == (0.0, 0.0, 0.0)


def test_snr_scaling():
    model = np.zeros((32, 32))
    model[10:20, 10:20] = 10.0  # 100 pixels of signal
    snr = estimate_snr(model, noise_sigma=1.0)
    assert snr == pytest.approx(1000 / np.sqrt(100))
    assert estimate_snr(model, noise_sigma=100.0) == 0.0  # nothing above threshold
    per_pixel = np.full(model.shape, 2.0)
    assert estimate_snr(model, per_pixel) == pytest.approx(1000 / np.sqrt(400))
