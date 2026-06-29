"""Core unfiltering science logic — bin lookup and polynomial regression application."""
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures

from prod.std.standard_method import CLOUD_VALUES, SCENE_TYPES

# TODO: replace with per-sample lookup from scene/cloud input file when it exists
HARDCODED_SCENE = "Clear Ocean"
HARDCODED_CLOUD = 0

_SZA_EDGES = [0.0, 22.2, 41.4, 60.0, 75.5, 85.0]
_VZA_EDGES = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
_RAZ_EDGES = [0.0, 15.0, 60.0, 120.0, 165.0, 180.0]


def load_coefficients(coef_path: Path | str) -> xr.Dataset:
    return xr.open_dataset(coef_path)


def _fold_raz(raz: np.ndarray) -> np.ndarray:
    """Fold RAZ from [0,360] to [0,180]."""
    return np.where(raz > 180, 360.0 - raz, raz)


def _cut(arr: np.ndarray, edges: list) -> np.ndarray:
    """Map arr values to bin indices (0–4), or -1 if out of range."""
    labels = pd.cut(arr, bins=edges, labels=False, include_lowest=True, right=True)
    result = np.asarray(labels, dtype=float)
    return np.where(np.isnan(result), -1, result).astype(int)


def _assign_bin_indices(sza: np.ndarray, vza: np.ndarray, raz: np.ndarray):
    """Return (sza_idx, vza_idx, raz_idx) arrays; -1 where angle is out of range."""
    return _cut(sza, _SZA_EDGES), _cut(vza, _VZA_EDGES), _cut(raz, _RAZ_EDGES)


def apply_unfiltering(
    sw_f: np.ndarray,
    ssw_f: np.ndarray,
    lw_f: np.ndarray,
    tot_f: np.ndarray,
    sza: np.ndarray,
    vza: np.ndarray,
    raz: np.ndarray,
    coef_ds: xr.Dataset,
    scene_idx: int,
    cloud: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply regression coefficients to produce unfiltered radiances.

    Returns
    -------
    (sw_u, ssw_u, lw_u, tot_u) — NaN where the bin has no coefficient data.
    """
    raz = _fold_raz(raz)
    sza_idx, vza_idx, raz_idx = _assign_bin_indices(sza, vza, raz)
    cloud_idx = CLOUD_VALUES.index(cloud)

    n = len(sw_f)
    sw_u  = np.full(n, np.nan, dtype=float)
    ssw_u = np.full(n, np.nan, dtype=float)
    lw_u  = np.full(n, np.nan, dtype=float)
    tot_u = np.full(n, np.nan, dtype=float)

    poly = PolynomialFeatures(degree=2, include_bias=True)

    sw_coefs  = coef_ds["sw_coefficients"].values[scene_idx, cloud_idx]   # (5,5,5,7)
    ssw_coefs = coef_ds["ssw_coefficients"].values[scene_idx, cloud_idx]
    lw_coefs  = coef_ds["lw_coefficients"].values[scene_idx, cloud_idx]   # (5,5,5,3)
    tot_coefs = coef_ds["tot_coefficients"].values[scene_idx, cloud_idx]

    for si, vi, ri in iproduct(range(5), range(5), range(5)):
        mask = (sza_idx == si) & (vza_idx == vi) & (raz_idx == ri)
        if not mask.any():
            continue

        sw_coef  = sw_coefs[si, vi, ri]
        ssw_coef = ssw_coefs[si, vi, ri]
        lw_coef  = lw_coefs[si, vi, ri]
        tot_coef = tot_coefs[si, vi, ri]

        if np.any(np.isnan(sw_coef)):
            continue  # sparse bin — leave NaN

        X_poly = poly.fit_transform(np.vstack([ssw_f[mask], sw_f[mask]]).T)
        sw_u[mask]  = sw_coef[0]  + X_poly @ sw_coef[1:]
        ssw_u[mask] = ssw_coef[0] + X_poly @ ssw_coef[1:]

        lw_u[mask]  = lw_coef[0]  + lw_coef[1] * lw_f[mask]  + lw_coef[2] * lw_f[mask]**2
        tot_u[mask] = tot_coef[0] + tot_coef[1] * tot_f[mask] + tot_coef[2] * tot_f[mask]**2

    return sw_u, ssw_u, lw_u, tot_u