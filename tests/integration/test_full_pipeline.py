"""
Full end-to-end pipeline integration test.

Covers three stages in sequence:
  1. Coefficient generation — load all MODTRAN 3.7 Tape7 files, fit regression, write .nc
  2. L2 algorithm run     — load L1B + FMATCH-CAM from manifest, classify scene/cloud,
                            apply unfiltering, write output NetCDF and manifest
  3. Output verification  — check structure, variables, and that some samples are filled

Run:   pytest -m integration tests/integration/test_full_pipeline.py -v
Skip:  pytest -m "not integration"

Fixtures are module-scoped so the expensive generation and algorithm steps run once.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_REPO = Path(__file__).parent.parent.parent
_DATA_37 = _REPO / "data" / "Modtran_3-7_data"
_SRF_DIR = _REPO / "data" / "SRF"
_MANIFEST = _REPO / "data" / "manifest" / "LIBERA_INPUT_MANIFEST_01KVRA2GC8R3NGRTE5J8X2P8PE.json"

pytestmark = pytest.mark.integration

_L2_VARIABLES = [
    "shortwave_unfiltered_radiance",
    "longwave_unfiltered_radiance",
    "split_shortwave_unfiltered_radiance",
    "total_unfiltered_radiance",
    "latitude",
    "longitude",
    "quality_flags",
]


# ---------------------------------------------------------------------------
# Stage 1 fixture — coefficient generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coef_file(tmp_path_factory):
    """Generate fresh coefficients from MODTRAN 3.7 data into a temp directory."""
    if not list(_DATA_37.rglob("*.tp7")):
        pytest.skip("No .tp7 files found under data/Modtran_3-7_data/")

    from prod.std.standard_method import run

    out_path = tmp_path_factory.mktemp("coef") / "unfiltering_coefficients_test.nc"
    return run(
        data_dir=_DATA_37,
        srf_dir=_SRF_DIR,
        srf_version="0-0-1",
        modtran_version="3.7",
        output_path=out_path,
    )


# ---------------------------------------------------------------------------
# Stage 2 fixture — algorithm run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory, coef_file):
    """Run the L2 algorithm and return the output directory and file paths."""
    if not _MANIFEST.exists():
        pytest.skip(f"Two-file manifest not found: {_MANIFEST}")

    from unfiltered_radiances.algorithm import algorithm

    out_dir = tmp_path_factory.mktemp("l2_output")
    prev_processing = os.environ.get("PROCESSING_PATH")
    prev_coef = os.environ.get("COEFFICIENTS_FILE")

    os.environ["PROCESSING_PATH"] = str(out_dir)
    os.environ["COEFFICIENTS_FILE"] = str(coef_file)

    try:
        algorithm(_MANIFEST)
    finally:
        if prev_processing is None:
            os.environ.pop("PROCESSING_PATH", None)
        else:
            os.environ["PROCESSING_PATH"] = prev_processing
        if prev_coef is None:
            os.environ.pop("COEFFICIENTS_FILE", None)
        else:
            os.environ["COEFFICIENTS_FILE"] = prev_coef

    return {
        "out_dir": out_dir,
        "nc_files": sorted(out_dir.glob("LIBERA_L2_*.nc")),
        "manifest_files": sorted(out_dir.glob("LIBERA_OUTPUT_MANIFEST_*.json")),
    }


# ---------------------------------------------------------------------------
# Stage 1 tests — coefficient file
# ---------------------------------------------------------------------------

class TestCoefficientGeneration:
    def test_file_exists(self, coef_file):
        assert coef_file.exists()

    def test_expected_variables(self, coef_file):
        with xr.open_dataset(coef_file) as ds:
            for var in ["sw_coefficients", "ssw_coefficients", "lw_coefficients", "tot_coefficients"]:
                assert var in ds.data_vars, f"Missing variable: {var}"

    def test_dimension_shapes(self, coef_file):
        with xr.open_dataset(coef_file) as ds:
            # SW/SSW use multi_coef_idx (7 params); LW/TOT use coef_idx (3 params)
            for var in ["sw_coefficients", "ssw_coefficients"]:
                v = ds[var]
                assert v.dims == ("scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "multi_coef_idx")
                assert v.shape == (5, 2, 5, 5, 5, 7)
            for var in ["lw_coefficients", "tot_coefficients"]:
                v = ds[var]
                assert v.dims == ("scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "coef_idx")
                assert v.shape == (5, 2, 5, 5, 5, 3)

    def test_some_bins_filled(self, coef_file):
        with xr.open_dataset(coef_file) as ds:
            sw = ds["sw_coefficients"].values
        assert np.isfinite(sw).any(), "All SW coefficient bins are NaN — no data was fitted"

    def test_global_attributes(self, coef_file):
        with xr.open_dataset(coef_file) as ds:
            assert "srf_version" in ds.attrs
            assert "modtran_version" in ds.attrs


# ---------------------------------------------------------------------------
# Stage 2+3 tests — L2 output files
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_exactly_one_nc_output(self, pipeline_output):
        assert len(pipeline_output["nc_files"]) == 1, (
            f"Expected 1 L2 NC file, found {len(pipeline_output['nc_files'])}"
        )

    def test_exactly_one_output_manifest(self, pipeline_output):
        assert len(pipeline_output["manifest_files"]) == 1, (
            f"Expected 1 output manifest, found {len(pipeline_output['manifest_files'])}"
        )

    def test_output_has_expected_variables(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            for var in _L2_VARIABLES:
                assert var in ds, f"Missing variable in output: {var}"

    def test_radiometer_time_coordinate_present(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            assert "RADIOMETER_TIME" in ds.dims

    def test_product_id_attribute(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            assert ds.attrs.get("ProductID") == "UNF-RAD"

    def test_some_samples_filled(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            sw = ds["shortwave_unfiltered_radiance"].values
        filled = int(np.isfinite(sw).sum())
        assert filled > 0, "shortwave_unfiltered_radiance is all NaN — no samples were filled"

    def test_output_sample_count_matches_l1b(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            n = ds.sizes["RADIOMETER_TIME"]
        # L1B example file has 602,200 samples
        assert n == 602_200, f"Expected 602200 samples, got {n}"

    def test_latitude_in_valid_range(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            lat = ds["latitude"].values
        assert np.nanmin(lat) >= -90 and np.nanmax(lat) <= 90

    def test_longitude_in_valid_range(self, pipeline_output):
        with xr.open_dataset(pipeline_output["nc_files"][0]) as ds:
            lon = ds["longitude"].values
        assert np.nanmin(lon) >= -180 and np.nanmax(lon) <= 180
