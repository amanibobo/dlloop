"""Cosmology helpers shared by the simulation backends.

Ported from ``deeplense/lens.py`` (DeepLenseSim), with one deliberate fix: DeepLenseSim computes
Einstein radii from *luminosity* distances. The point-mass Einstein radius is

    theta_E = sqrt( 4 G M / c^2  *  D_ls / (D_l D_s) )

with *angular-diameter* distances. For a 1e12 M_sun lens at z=0.5 and a source at z=1 this is the
difference between ~1.35" (DeepLenseSim) and ~1.66" (correct).
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.constants import G, M_sun, c
from astropy.cosmology import FlatLambdaCDM

RAD_TO_ARCSEC = 206264.806

DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)


def point_mass_einstein_radius(
    mass_msun: float | np.ndarray,
    z_lens: float,
    z_source: float,
    cosmo: FlatLambdaCDM = DEFAULT_COSMOLOGY,
) -> float | np.ndarray:
    """Einstein radius in arcsec of a point mass (or array of point masses) in M_sun."""
    if z_source <= z_lens:
        raise ValueError("source must be behind the lens (z_source > z_lens)")
    d_l = cosmo.angular_diameter_distance(z_lens).to(u.m)
    d_s = cosmo.angular_diameter_distance(z_source).to(u.m)
    try:  # astropy >= 7 accepts two redshifts directly; older versions need _z1z2
        d_ls = cosmo.angular_diameter_distance(z_lens, z_source).to(u.m)
    except TypeError:
        d_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m)
    m = np.asarray(mass_msun, dtype=float) * M_sun
    theta = np.sqrt(4 * G * m / c**2 * d_ls / (d_l * d_s))
    out = (theta.to(u.dimensionless_unscaled).value) * RAD_TO_ARCSEC
    return float(out) if np.ndim(out) == 0 else out


def axion_mass_to_vortex_length(axion_mass_ev: float) -> float:
    """Half-length of the axion vortex line in arcsec, from the axion mass in eV.

    Direct port of DeepLenseSim's ``axion_mass_to_length`` (which is the same expression as its
    ``axion_length_to_mass``). DeepLenseSim notes the calibration is only valid for a z=0.5 halo.
    1e-24 eV -> 3.0", 1e-22 eV -> 0.03".
    """
    return 0.06 / (2.0 * float(axion_mass_ev)) * 1e-22
