"""
Generates unfiltering coefficients for the traditional regression method.

Workflow:
  1. load_dataset()                      — parse all .tp7 files into a single DataFrame
  2. generate_unfiltering_coefficients() — fit regression per SZA/VZA/RAZ bin
  3. serialize_coefficients()            — write the result to a NetCDF file

Run end-to-end:
  from prod.std.standard_method import run
  run(data_dir="data/Modtran_3-7_data/", srf_dir="data/SRF/", srf_version="0-0-1", modtran_version="3.7")
"""

import logging
import subprocess
import tomllib  # stdlib in Python 3.11+
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tp7.tp7 import Tape7, _as_path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent
_COEFFICIENTS_DIR = _REPO_ROOT / "coefficients"

# Viewing geometry bins (degrees) — consistent with Loeb et al. (2001)
SZA_BINS = [
    (0.0, 22.2),
    (22.2, 41.4),
    (41.4, 60.0),
    (60.0, 75.5),
    (75.5, 85.0),
]

VZA_BINS = [
    (0.0, 15.0),
    (15.0, 30.0),
    (30.0, 45.0),
    (45.0, 60.0),
    (60.0, 90.0),
]

RAZ_BINS = [
    (0.0, 15.0),
    (15.0, 60.0),
    (60.0, 120.0),
    (120.0, 165.0),
    (165.0, 180.0),
]

SCENE_TYPES = ["Land", "Cloudy Ocean", "Clear Ocean", "Snow", "Deep Convective Cloud"]
CLOUD_VALUES = [0, 1]


def _code_version() -> str:
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_output_filename(srf_version: str, modtran_version: str) -> Path:
    version = _code_version()
    return _COEFFICIENTS_DIR / (
        f"unfiltering_coefficients"
        f"_v{version}"
        f"_srf-{srf_version}"
        f"_modtran-{modtran_version}"
        f".nc"
    )


def load_dataset(data_dir, srf_dir=None) -> pd.DataFrame:
    """
    Parse every .tp7 file found under data_dir and return a single concatenated
    describer_df with integrated filtered and unfiltered radiances.

    Parameters
    ----------
    data_dir : str | Path | S3Path
        Root directory containing the MODTRAN .tp7 files (searched recursively).
    srf_dir : str | Path | S3Path, optional
        Directory containing the Libera SRF CSV files.  Defaults to the local
        data/SRF/ directory relative to the repo root.

    Returns
    -------
    pd.DataFrame
        Combined dataset with one row per MODTRAN run across all scene types.
    """
    #TODO update to use .nc files from S3 
    #TODO look into cloudpathlib to handle both local and S3 paths with the same code class is called AnyPath
    data_path = _as_path(data_dir)
    tp7_files = sorted(data_path.rglob("*.tp7"))

    if not tp7_files:
        raise FileNotFoundError(f"No .tp7 files found under {data_dir}")

    frames = []
    #TODO build functionality into parser to handle nc files as well instead of just .tp7 files
    for tp7_file in tp7_files:
        logger.info(f"Loading {tp7_file.name}")
        t7 = Tape7(tp7_file, srf_path=srf_dir)
        frames.append(t7.describer_df)

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(dataset)} runs from {len(tp7_files)} files")
    return dataset


def load_nc_dataset(data_dir, srf_dir=None) -> pd.DataFrame:
    """
    Parse every MODTRAN 6 .nc file under data_dir using Modtran6NC and return
    a single concatenated describer_df with integrated filtered and unfiltered
    radiances — same column contract as load_dataset().

    Parameters
    ----------
    data_dir : str | Path | S3Path
        Root directory containing the MODTRAN 6 .nc files (searched recursively).
    srf_dir : str | Path | S3Path, optional
        Directory containing the Libera SRF CSV files.  Defaults to the local
        data/SRF/ directory relative to the repo root.

    Returns
    -------
    pd.DataFrame
        Combined dataset with one row per angle combination across all .nc files.
    """
    from tp7.modtran6 import Modtran6NC

    data_path = _as_path(data_dir)
    nc_files = sorted(data_path.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No .nc files found under {data_dir}")

    frames = []
    for nc_file in nc_files:
        logger.info(f"Loading {nc_file.name}")
        m6 = Modtran6NC(nc_file, srf_path=srf_dir)
        frames.append(m6.describer_df)

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(dataset)} rows from {len(nc_files)} files")
    return dataset


def generate_unfiltering_coefficients(dataset: pd.DataFrame) -> dict:
    """
    Fit quadratic regression coefficients per scene/cloud/SZA/VZA/RAZ bin.

    SW and SSW use a multivariate degree-2 polynomial with SSW and SW filtered
    radiances as predictors:
        m_u = c0 + c1*1 + c2*ssw_f + c3*sw_f + c4*ssw_f^2 + c5*ssw_f*sw_f + c6*sw_f^2

    LW and TOT use a univariate degree-2 polynomial:
        m_u = a0 + a1*m_f + a2*m_f^2

    Parameters
    ----------
    dataset : pd.DataFrame
        Output of load_dataset() — must contain Scene, Cloud, and all integrated
        radiance columns including Total Unfiltered Rads (Integrated).

    Returns
    -------
    dict
        Keys are (scene_idx, cloud, sza_bin, vza_bin, raz_bin) tuples where
        scene_idx is an index into SCENE_TYPES and cloud is 0 or 1.
        Values are (sw_coef, ssw_coef, lw_coef, tot_coef) where:
          sw_coef  : ndarray shape (7,)  — multivariate [intercept, c1..c6]
          ssw_coef : ndarray shape (7,)  — multivariate [intercept, c1..c6]
          lw_coef  : ndarray shape (3,)  — univariate [a0, a1, a2]
          tot_coef : ndarray shape (3,)  — univariate [a0, a1, a2]
          Any of the four may be None if the bin contains insufficient data.
    """
    coefficients = {}

    angular_bins = [
        (sza_bin, vza_bin, raz_bin)
        for sza_bin in SZA_BINS
        for vza_bin in VZA_BINS
        for raz_bin in RAZ_BINS
    ]

    total_bins = len(SCENE_TYPES) * len(CLOUD_VALUES) * len(angular_bins)

    for scene_idx, scene_name in enumerate(SCENE_TYPES):
        for cloud in CLOUD_VALUES:
            scene_cloud_data = dataset[
                (dataset["Scene"] == scene_name) &
                (dataset["Cloud"] == cloud)
            ]

            for sza_bin, vza_bin, raz_bin in angular_bins:
                binned = scene_cloud_data[
                    (scene_cloud_data["SZA"].between(*sza_bin)) &
                    (scene_cloud_data["VZA"].between(*vza_bin)) &
                    (scene_cloud_data["RAZ"].between(*raz_bin))
                ]

                sw_coef = ssw_coef = lw_coef = tot_coef = None

                if len(binned) >= 3:
                    try:
                        filtered_sw    = binned["Shortwave Filtered Rads (Integrated)"]
                        unfiltered_sw  = binned["Shortwave Unfiltered Rads (Integrated)"]
                        filtered_lw    = binned["Longwave Filtered Rads (Integrated)"]
                        unfiltered_lw  = binned["Longwave Unfiltered Rads (Integrated)"]
                        filtered_ssw   = binned["Split Shortwave Filtered Rads (Integrated)"]
                        unfiltered_ssw = binned["Split Shortwave Unfiltered Rads (Integrated)"]
                        filtered_tot   = binned["Total Filtered Rads (Integrated)"]
                        unfiltered_tot = binned["Total Unfiltered Rads (Integrated)"]

                        fit_lw = np.polynomial.polynomial.Polynomial.fit(filtered_lw, unfiltered_lw, 2)
                        lw_coef = fit_lw.convert().coef

                        fit_tot = np.polynomial.polynomial.Polynomial.fit(filtered_tot, unfiltered_tot, 2)
                        tot_coef = fit_tot.convert().coef

                        X = np.vstack([filtered_ssw, filtered_sw]).T
                        poly = PolynomialFeatures(degree=2)
                        X_poly = poly.fit_transform(X)

                        reg_sw = LinearRegression()
                        reg_sw.fit(X_poly, unfiltered_sw.values)
                        sw_coef = np.concatenate([[reg_sw.intercept_], reg_sw.coef_])

                        reg_ssw = LinearRegression()
                        reg_ssw.fit(X_poly, unfiltered_ssw.values)
                        ssw_coef = np.concatenate([[reg_ssw.intercept_], reg_ssw.coef_])

                    except Exception as e:
                        logger.warning(
                            f"Fit failed for bin ({scene_name}, cloud={cloud}, "
                            f"{(sza_bin, vza_bin, raz_bin)}): {e}"
                        )
                else:
                    logger.debug(
                        f"Skipping sparse bin ({scene_name}, cloud={cloud}, "
                        f"{(sza_bin, vza_bin, raz_bin)}) ({len(binned)} samples)"
                    )

                coefficients[(scene_idx, cloud, sza_bin, vza_bin, raz_bin)] = (
                    sw_coef, ssw_coef, lw_coef, tot_coef
                )

    filled = sum(1 for v in coefficients.values() if v[0] is not None)
    logger.info(f"Coefficients generated for {filled}/{total_bins} bins")
    return coefficients


def serialize_coefficients(
    coefficients: dict,
    output_path,
    srf_version: str = "unknown",
    modtran_version: str = "unknown",
) -> Path:
    """
    Write the coefficient dict to a NetCDF file.

    Dimensions
    ----------
    scene        : 0–4  — index into SCENE_TYPES
    cloud        : 0–1  — binary cloud flag (0=no cloud, 1=any cloud)
    sza_bin, vza_bin, raz_bin : int (0–4)
        Bin indices.  Corresponding angle ranges stored as auxiliary
        coordinates sza_lo/sza_hi, vza_lo/vza_hi, raz_lo/raz_hi.
    sw_coef_idx  : 0–2  — polynomial order [a0, a1, a2]
    lw_coef_idx  : 0–2  — polynomial order [a0, a1, a2]
    tot_coef_idx : 0–2  — polynomial order [a0, a1, a2]
    ssw_coef_idx : 0–6  — [intercept, 1, ssw_f, sw_f, ssw_f^2, ssw_f*sw_f, sw_f^2]

    Sparse bins (no data) are stored as NaN.

    Parameters
    ----------
    coefficients : dict
        Output of generate_unfiltering_coefficients().
    output_path : str | Path
        Destination path for the .nc file.
    srf_version : str
        SRF version string recorded in the file's global attributes.
    modtran_version : str
        MODTRAN version string recorded in the file's global attributes.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    n_scene = len(SCENE_TYPES)
    n_cloud = len(CLOUD_VALUES)
    n_sza, n_vza, n_raz = len(SZA_BINS), len(VZA_BINS), len(RAZ_BINS)

    sw_arr  = np.full((n_scene, n_cloud, n_sza, n_vza, n_raz, 7), np.nan)
    ssw_arr = np.full((n_scene, n_cloud, n_sza, n_vza, n_raz, 7), np.nan)
    lw_arr  = np.full((n_scene, n_cloud, n_sza, n_vza, n_raz, 3), np.nan)
    tot_arr = np.full((n_scene, n_cloud, n_sza, n_vza, n_raz, 3), np.nan)

    for (scene_idx, cloud, sza_bin, vza_bin, raz_bin), (sw_coef, ssw_coef, lw_coef, tot_coef) in coefficients.items():
        i = SZA_BINS.index(sza_bin)
        j = VZA_BINS.index(vza_bin)
        k = RAZ_BINS.index(raz_bin)
        c = CLOUD_VALUES.index(cloud)
        if sw_coef is not None:
            sw_arr[scene_idx, c, i, j, k]  = sw_coef
        if ssw_coef is not None:
            ssw_arr[scene_idx, c, i, j, k] = ssw_coef
        if lw_coef is not None:
            lw_arr[scene_idx, c, i, j, k]  = lw_coef
        if tot_coef is not None:
            tot_arr[scene_idx, c, i, j, k] = tot_coef

    _MULTI_DESC = (
        "Multivariate degree-2 polynomial using filtered SSW and SW radiances as inputs.  "
        "Prediction: m_u = c0 + c1*1 + c2*ssw_f + c3*sw_f + c4*ssw_f^2 + c5*ssw_f*sw_f + c6*sw_f^2  "
        "where ssw_f and sw_f are the filtered split-shortwave and shortwave radiances.  "
        "multi_coef_idx mapping:  "
        "  0 = intercept (LinearRegression bias term);  "
        "  1 = c1, weight for the constant feature from PolynomialFeatures (near 0, redundant with intercept);  "
        "  2 = c2, weight for ssw_f (linear SSW term);  "
        "  3 = c3, weight for sw_f (linear SW term);  "
        "  4 = c4, weight for ssw_f^2 (quadratic SSW term);  "
        "  5 = c5, weight for ssw_f*sw_f (cross term);  "
        "  6 = c6, weight for sw_f^2 (quadratic SW term).  "
        "NaN indicates the bin had fewer than 3 samples and no fit was attempted."
    )
    _POLY_DESC = (
        "Univariate quadratic polynomial: m_u = a0 + a1*m_f + a2*m_f^2  "
        "where m_f is the filtered radiance and m_u is the unfiltered radiance.  "
        "coef_idx mapping:  "
        "  0 = a0 (constant offset, W m-2 sr-1);  "
        "  1 = a1 (linear scaling factor, dimensionless, expected near 1.0);  "
        "  2 = a2 (quadratic correction, W-1 m2 sr, typically very small).  "
        "NaN indicates the bin had fewer than 3 samples and no fit was attempted."
    )

    dims_multi = ["scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "multi_coef_idx"]
    dims_poly  = ["scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "coef_idx"]

    ds = xr.Dataset(
        {
            "sw_coefficients": (
                dims_multi,
                sw_arr,
                {"long_name": "Shortwave unfiltering multivariate polynomial coefficients", "description": _MULTI_DESC, "units": "W m-2 sr-1"},
            ),
            "ssw_coefficients": (
                dims_multi,
                ssw_arr,
                {"long_name": "Split-shortwave unfiltering multivariate polynomial coefficients", "description": _MULTI_DESC, "units": "W m-2 sr-1"},
            ),
            "lw_coefficients": (
                dims_poly,
                lw_arr,
                {"long_name": "Longwave unfiltering polynomial coefficients", "description": _POLY_DESC, "units": "W m-2 sr-1"},
            ),
            "tot_coefficients": (
                dims_poly,
                tot_arr,
                {"long_name": "Total channel unfiltering polynomial coefficients", "description": _POLY_DESC, "units": "W m-2 sr-1"},
            ),
        },
        coords={
            "scene": ("scene", SCENE_TYPES, {
                "long_name": "Scene type",
                "description": (
                    "0 = Land: non-ocean, non-snow/ice surface (all IGBP types except 15 and 17); "
                    "1 = Cloudy Ocean: IGBP 17 (Water Bodies) with cloud_fraction > 0.10; "
                    "2 = Clear Ocean: IGBP 17 (Water Bodies) with cloud_fraction <= 0.10; "
                    "3 = Snow/Ice: IGBP 15 (Snow and Ice) or NISE permanent ice or dry snow; "
                    "4 = Deep Convective Cloud: cloud_fraction >= 0.10 and cloud_optical_thickness > 10.0 (any surface)."
                ),
            }),
            "cloud":          ("cloud", CLOUD_VALUES, {"long_name": "Cloud binary flag (0=no cloud, 1=any cloud)"}),
            "sza_bin":        ("sza_bin", range(n_sza), {"long_name": "Solar Zenith Angle bin index (0–4); see sza_lo/sza_hi for degree ranges"}),
            "vza_bin":        ("vza_bin", range(n_vza), {"long_name": "Viewing Zenith Angle bin index (0–4); see vza_lo/vza_hi for degree ranges"}),
            "raz_bin":        ("raz_bin", range(n_raz), {"long_name": "Relative Azimuth Angle bin index (0–4); see raz_lo/raz_hi for degree ranges"}),
            "multi_coef_idx": ("multi_coef_idx", range(7), {"long_name": "Multivariate coefficient index: 0=intercept, 1=const, 2=ssw_f, 3=sw_f, 4=ssw_f^2, 5=ssw_f*sw_f, 6=sw_f^2"}),
            "coef_idx":       ("coef_idx", range(3), {"long_name": "Univariate polynomial coefficient index: 0=a0 (offset), 1=a1 (linear), 2=a2 (quadratic)"}),
            "sza_lo": ("sza_bin", [b[0] for b in SZA_BINS], {"long_name": "SZA bin lower bound, inclusive (degrees)"}),
            "sza_hi": ("sza_bin", [b[1] for b in SZA_BINS], {"long_name": "SZA bin upper bound, inclusive (degrees)"}),
            "vza_lo": ("vza_bin", [b[0] for b in VZA_BINS], {"long_name": "VZA bin lower bound, inclusive (degrees)"}),
            "vza_hi": ("vza_bin", [b[1] for b in VZA_BINS], {"long_name": "VZA bin upper bound, inclusive (degrees)"}),
            "raz_lo": ("raz_bin", [b[0] for b in RAZ_BINS], {"long_name": "RAZ bin lower bound, inclusive (degrees)"}),
            "raz_hi": ("raz_bin", [b[1] for b in RAZ_BINS], {"long_name": "RAZ bin upper bound, inclusive (degrees)"}),
        },
        attrs={
            "title": "Libera unfiltering regression coefficients",
            "method": "Scene/cloud-stratified quadratic regression (Loeb et al. 2001)",
            "bin_convention": "All angle bins are inclusive on both lower and upper bounds.",
            "coefficient_version": _code_version(),
            "srf_version": srf_version,
            "modtran_version": modtran_version,
            "git_commit": _git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    output_path = Path(output_path)
    ds.to_netcdf(output_path)
    logger.info(f"Coefficients written to {output_path}")
    return output_path


def run(
    data_dir,
    srf_dir,
    srf_version: str,
    modtran_version: str,
    output_path=None,
) -> Path:
    """
    End-to-end coefficient generation: load data, fit, and write .nc file.

    Parameters
    ----------
    data_dir : str | Path | S3Path
        Root directory containing MODTRAN .tp7 files.
    srf_dir : str | Path | S3Path
        Directory containing the Libera SRF CSV files.
    srf_version : str
        SRF version string, e.g. "0-0-1".  Recorded in the output filename
        and .nc global attributes.
    modtran_version : str
        MODTRAN version string, e.g. "3.7".  Recorded in the .nc global
        attributes.
    output_path : str | Path, optional
        Destination for the .nc file.  Defaults to
        coefficients/unfiltering_coefficients_v{code}_srf-{srf}_modtran-{modtran}.nc

    Returns
    -------
    Path
        Path to the written coefficients file.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if output_path is None:
        output_path = _build_output_filename(srf_version, modtran_version)

    logger.info("Step 1: Loading dataset")
    # TODO: dispatch to load_nc_dataset() when modtran_version starts with "6";
    dataset = load_dataset(data_dir, srf_dir=srf_dir)

    logger.info("Step 2: Generating coefficients")
    coefficients = generate_unfiltering_coefficients(dataset)

    logger.info("Step 3: Serializing to NetCDF")
    return serialize_coefficients(
        coefficients,
        output_path,
        srf_version=srf_version,
        modtran_version=modtran_version,
    )
