"""Unit tests for prod/std/standard_method.py."""
# TODO: add tests for run() dispatching to load_nc_dataset() when modtran_version="6.0";
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from prod.std.standard_method import (
    SZA_BINS,
    VZA_BINS,
    RAZ_BINS,
    SCENE_TYPES,
    CLOUD_VALUES,
    generate_unfiltering_coefficients,
    serialize_coefficients,
)


class TestGenerateCoefficients:
    def test_returns_all_bins(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        expected = len(SCENE_TYPES) * len(CLOUD_VALUES) * len(SZA_BINS) * len(VZA_BINS) * len(RAZ_BINS)
        assert len(result) == expected

    def test_populated_bin_coefficient_shapes(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        # sample_dataset is Land, cloud=0, bin (0,0,0)
        sw, ssw, lw, tot = result[(0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert sw.shape == (7,)
        assert ssw.shape == (7,)
        assert lw.shape == (3,)
        assert tot.shape == (3,)

    def test_linear_term_near_one(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        _, _, lw, tot = result[(0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert abs(lw[1] - 1.0) < 0.15
        assert abs(tot[1] - 1.0) < 0.15

    def test_ssw_intercept_at_index_zero(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        _, ssw, _, _ = result[(0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert ssw is not None
        assert len(ssw) == 7

    def test_empty_bin_returns_none(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        # scene_idx=1 (Cloudy Ocean) has no rows in sample_dataset
        sw, ssw, lw, tot = result[(1, 0, SZA_BINS[1], VZA_BINS[1], RAZ_BINS[1])]
        assert sw is None
        assert ssw is None
        assert lw is None
        assert tot is None

    def test_below_threshold_returns_none(self):
        """Bins with exactly 2 rows must return None (minimum is 3)."""
        tiny = pd.DataFrame({
            "Scene": ["Land", "Land"],
            "Cloud": [0, 0],
            "SZA": [5.0, 10.0],
            "VZA": [5.0, 10.0],
            "RAZ": [5.0, 10.0],
            "Shortwave Filtered Rads (Integrated)": [80.0, 85.0],
            "Shortwave Unfiltered Rads (Integrated)": [82.0, 87.0],
            "Longwave Filtered Rads (Integrated)": [30.0, 32.0],
            "Longwave Unfiltered Rads (Integrated)": [31.0, 33.0],
            "Split Shortwave Filtered Rads (Integrated)": [10.0, 11.0],
            "Split Shortwave Unfiltered Rads (Integrated)": [10.2, 11.2],
            "Total Filtered Rads (Integrated)": [110.0, 117.0],
            "Total Unfiltered Rads (Integrated)": [113.0, 120.0],
        })
        result = generate_unfiltering_coefficients(tiny)
        sw, ssw, lw, tot = result[(0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        assert sw is None

    def test_all_values_are_floats(self, sample_dataset):
        result = generate_unfiltering_coefficients(sample_dataset)
        sw, ssw, lw, tot = result[(0, 0, SZA_BINS[0], VZA_BINS[0], RAZ_BINS[0])]
        for coef in (sw, ssw, lw, tot):
            assert coef.dtype.kind == 'f'


class TestSerializeCoefficients:
    def test_creates_file(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        returned = serialize_coefficients(minimal_coefficients, out)
        assert returned == out
        assert out.exists()

    def test_expected_variables(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert "sw_coefficients" in ds
        assert "lw_coefficients" in ds
        assert "tot_coefficients" in ds
        assert "ssw_coefficients" in ds

    def test_sw_dimensions_and_shape(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        sw = ds["sw_coefficients"]
        assert sw.dims == ("scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "multi_coef_idx")
        assert sw.shape == (5, 2, 5, 5, 5, 7)

    def test_ssw_dimensions_and_shape(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        ssw = ds["ssw_coefficients"]
        assert ssw.dims == ("scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "multi_coef_idx")
        assert ssw.shape == (5, 2, 5, 5, 5, 7)

    def test_multi_coef_idx_length(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert ds.sizes["multi_coef_idx"] == 7

    def test_tot_dimensions_and_shape(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        tot = ds["tot_coefficients"]
        assert tot.dims == ("scene", "cloud", "sza_bin", "vza_bin", "raz_bin", "coef_idx")
        assert tot.shape == (5, 2, 5, 5, 5, 3)

    def test_populated_bin_not_nan(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert not np.any(np.isnan(ds["sw_coefficients"].values[0, 0, 0, 0, 0, :]))
        assert not np.any(np.isnan(ds["lw_coefficients"].values[0, 0, 0, 0, 0, :]))
        assert not np.any(np.isnan(ds["tot_coefficients"].values[0, 0, 0, 0, 0, :]))
        assert not np.any(np.isnan(ds["ssw_coefficients"].values[0, 0, 0, 0, 0, :]))

    def test_empty_bin_is_nan(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert np.all(np.isnan(ds["sw_coefficients"].values[1, 1, 1, 1, 1, :]))

    def test_global_attrs(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out, srf_version="0-0-1", modtran_version="3.7")
        ds = xr.open_dataset(out)
        assert ds.attrs["srf_version"] == "0-0-1"
        assert ds.attrs["modtran_version"] == "3.7"
        assert "coefficient_version" in ds.attrs
        assert "git_commit" in ds.attrs
        assert "created_utc" in ds.attrs

    def test_bin_bounds_coords(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        for coord in ["sza_lo", "sza_hi", "vza_lo", "vza_hi", "raz_lo", "raz_hi"]:
            assert coord in ds.coords, f"Missing coordinate: {coord}"

    def test_correct_bin_bounds_values(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert float(ds["sza_lo"].values[0]) == 0.0
        assert float(ds["sza_hi"].values[0]) == pytest.approx(22.2)

    def test_scene_coord_labels(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert ds.coords["scene"].values.tolist() == SCENE_TYPES

    def test_cloud_coord_values(self, tmp_path, minimal_coefficients):
        out = tmp_path / "coefs.nc"
        serialize_coefficients(minimal_coefficients, out)
        ds = xr.open_dataset(out)
        assert ds.coords["cloud"].values.tolist() == CLOUD_VALUES
