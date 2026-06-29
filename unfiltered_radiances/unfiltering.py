"""Core unfiltering science logic — scene/cloud classification, bin lookup, and polynomial regression."""
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import PolynomialFeatures

from prod.std.standard_method import CLOUD_VALUES, SCENE_TYPES

_SZA_EDGES = [0.0, 22.2, 41.4, 60.0, 75.5, 85.0]
_VZA_EDGES = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
_RAZ_EDGES = [0.0, 15.0, 60.0, 120.0, 165.0, 180.0]

_CLOUD_FRACTION_THRESHOLD = 0.10
_DCC_OT_THRESHOLD = 10.0
_IGBP_OCEAN = 17
_IGBP_SNOW = 15

_IDX_LAND    = SCENE_TYPES.index("Land")
_IDX_CLO_OCE = SCENE_TYPES.index("Cloudy Ocean")
_IDX_CLR_OCE = SCENE_TYPES.index("Clear Ocean")
_IDX_SNOW    = SCENE_TYPES.index("Snow")
_IDX_DCC     = SCENE_TYPES.index("Deep Convective Cloud")


def load_coefficients(coef_path: Path | str) -> xr.Dataset:
    return xr.open_dataset(coef_path)


def classify_scene_cloud(cam_ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive per-sample scene index and binary cloud flag from a FMATCH-CAM dataset.

    Returns
    -------
    scene_idx : np.ndarray[int], shape (n,)
        Index into SCENE_TYPES for each sample.
    cloud : np.ndarray[int], shape (n,)
        0 = no cloud, 1 = cloud present (cloud_fraction > 0.10).
    """
    igbp      = cam_ds["igbp_surface_type"].values.astype(int)
    cf        = cam_ds["viirs_cloud_cloud_fraction"].values
    cot       = cam_ds["viirs_cloud_cloud_optical_thickness"].values
    perm_ice  = cam_ds["nise_permanent_ice"].values
    dry_snow  = cam_ds["nise_dry_snow_on_land"].values

    n = len(igbp)
    scene_idx = np.full(n, _IDX_LAND, dtype=int)
    cloud     = (cf > _CLOUD_FRACTION_THRESHOLD).astype(int)

    # Priority 1: DCC
    dcc = (cf >= _CLOUD_FRACTION_THRESHOLD) & (cot > _DCC_OT_THRESHOLD)
    scene_idx[dcc] = _IDX_DCC
    cloud[dcc] = 1

    # Priority 2: Snow (not already DCC)
    snow = ~dcc & ((igbp == _IGBP_SNOW) | (perm_ice > 0.5) | (dry_snow > 0.5))
    scene_idx[snow] = _IDX_SNOW

    # Priority 3: Ocean (not DCC or Snow)
    ocean = ~dcc & ~snow & (igbp == _IGBP_OCEAN)
    scene_idx[ocean & (cf > _CLOUD_FRACTION_THRESHOLD)] = _IDX_CLO_OCE
    scene_idx[ocean & (cf <= _CLOUD_FRACTION_THRESHOLD)] = _IDX_CLR_OCE

    # Priority 4: Land — already the default in scene_idx

    return scene_idx, cloud


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
    scene_idx: np.ndarray,
    cloud: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply regression coefficients to produce unfiltered radiances.

    Parameters
    ----------
    scene_idx : np.ndarray[int], shape (n,)
        Per-sample index into SCENE_TYPES.
    cloud : np.ndarray[int], shape (n,)
        Per-sample binary cloud flag (0 or 1).

    Returns
    -------
    (sw_u, ssw_u, lw_u, tot_u) — NaN where the bin has no coefficient data.
    """
    raz = _fold_raz(raz)
    sza_idx, vza_idx, raz_idx = _assign_bin_indices(sza, vza, raz)

    n = len(sw_f)
    sw_u  = np.full(n, np.nan, dtype=float)
    ssw_u = np.full(n, np.nan, dtype=float)
    lw_u  = np.full(n, np.nan, dtype=float)
    tot_u = np.full(n, np.nan, dtype=float)

    poly = PolynomialFeatures(degree=2, include_bias=True)

    sw_coefs  = coef_ds["sw_coefficients"].values   # (5, 2, 5, 5, 5, 7)
    ssw_coefs = coef_ds["ssw_coefficients"].values
    lw_coefs  = coef_ds["lw_coefficients"].values   # (5, 2, 5, 5, 5, 3)
    tot_coefs = coef_ds["tot_coefficients"].values

    for sci, cli, si, vi, ri in iproduct(range(5), range(2), range(5), range(5), range(5)):
        mask = (
            (scene_idx == sci) & (cloud == cli) &
            (sza_idx == si) & (vza_idx == vi) & (raz_idx == ri)
        )
        if not mask.any():
            continue

        sw_coef  = sw_coefs[sci, cli, si, vi, ri]
        ssw_coef = ssw_coefs[sci, cli, si, vi, ri]
        lw_coef  = lw_coefs[sci, cli, si, vi, ri]
        tot_coef = tot_coefs[sci, cli, si, vi, ri]

        if np.any(np.isnan(sw_coef)):
            continue  # sparse bin — leave NaN

        # SW and SSW: multivariate quadratic on [ssw_f, sw_f] — 7 coefficients each
        X_poly = poly.fit_transform(np.vstack([ssw_f[mask], sw_f[mask]]).T)
        sw_u[mask]  = sw_coef[0]  + X_poly @ sw_coef[1:]
        ssw_u[mask] = ssw_coef[0] + X_poly @ ssw_coef[1:]

        # LW and TOT: univariate quadratic — 3 coefficients each
        lw_u[mask]  = lw_coef[0]  + lw_coef[1]  * lw_f[mask]  + lw_coef[2]  * lw_f[mask]**2
        tot_u[mask] = tot_coef[0] + tot_coef[1] * tot_f[mask] + tot_coef[2] * tot_f[mask]**2

    return sw_u, ssw_u, lw_u, tot_u
