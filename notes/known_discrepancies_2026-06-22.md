# Known Discrepancies and Limitations — 2026-06-22

This document captures all known gaps between the current implementation and production-ready
behavior, including synthetic data limitations, coefficient coverage gaps, placeholder values,
duplicated constants, and missing features.

---

## 1. Synthetic Example Data Limitations

### CAM file (`data/example_data/LIBERA_ANC_FMATCH-CAM_V0-1-0_...nc`)

- Covers only **1,000 samples over 10 seconds**; the L1B covers 602,200 samples over ~1.5 hours.
- Angles (`solar_zenith_angle`, `viewing_zenith_angle`, `relative_azimuth_angle`) contain
  **physically impossible values** (e.g., VZA > 90°, SZA > 170°) — purely random synthetic data.
- NISE ice fractions are uniformly ~0.5, cloud fractions are uniformly distributed 0–1, and
  IGBP types are uniformly distributed 1–20.
- This causes nearly all 602,200 L1B samples to be classified as **Snow** (NISE permanent ice > 0.5
  fires the snow mask), leaving only ~12 CAM samples to map to Clear/Cloudy Ocean bins.

**Result:** Only 163,541 / 602,200 L1B samples get filled — not because the algorithm is wrong,
but because the synthetic CAM data maps almost nothing to bins with trained coefficients (which
currently only cover Clear and Cloudy Ocean). In production, real CAM data will distribute
samples across all 5 scene types and filling will increase substantially.

### L1B file (`data/example_data/LIBERA_L1B_RAD-4CH_V0-5-5_...nc`)

- Values are **not representative** of real Libera measurements (confirmed by team 2026-06-19).
- SW filtered radiance values (~3200 W m⁻² sr⁻¹ nm⁻¹) are far outside the training range.
- Used only for structural/pipeline testing, not for scientific validation.

### Time alignment (`algorithm.py:166–167`)

- Uses `method="nearest"` to align CAM to L1B times because the example files have mismatched
  RADIOMETER_TIME arrays (10 seconds vs 1.5 hours).
- In production, both files will share identical RADIOMETER_TIME values; nearest-neighbor will
  be a no-op but is kept for robustness.

---

## 2. Coefficient Coverage Gaps

### Only Clear Ocean and Cloudy Ocean have trained coefficients

- `coefficients/unfiltering_coefficients_v0.1.0_srf-0-0-1_modtran-3.7.nc` has non-NaN values
  only for scene indices 1 (Cloudy Ocean) and 2 (Clear Ocean).
- Land, Snow, and Deep Convective Cloud bins are all NaN → samples in those scenes produce NaN
  unfiltered radiances.
- **Fix:** Generate coefficients from the full MODTRAN 6 AWS dataset, which includes all scene types.

### `CERES_SCENE_MAP` in `tp7/modtran6.py:25–28` is incomplete

- Only scene IDs 5 (Clear Ocean) and 18 (Cloudy Ocean) are mapped.
- Land, Snow, and DCC scene IDs in MODTRAN 6 files are not yet known; encountering an unknown
  ID raises `ValueError` and halts processing.
- **Fix:** Extend the map as new M6 files with those scenes become available.

### Sparse angular bins

- Bins with < 3 training samples are skipped (`standard_method.py:221`) and stored as NaN.
- Sparse geometries (e.g., very high SZA + high VZA combinations) silently produce NaN output
  with no distinction from other NaN causes.

---

## 3. Unit and Value Range Issues

### Product definition YAML has wrong units (`l2-unfiltered-radiance-product-definition.yml:35–62`)

- All radiance variables are labeled `W/(m^2*sr*nm)` (spectral) but the outputs are broadband
  **integrated** values — units should be `W m⁻² sr⁻¹`.
- `valid_range: [0, 1000]` is a placeholder; real ranges are unknown until MODTRAN 6 coefficients
  are generated and validated against real L1B data.

### Training vs L1B unit scale mismatch (not a current blocker)

- MODTRAN 3.7 training radiances: `W sr⁻¹ cm⁻²`, SW range ~4–481.
- L1B example file SW values: ~3200 (unit label unclear, values not representative).
- Team confirmed 2026-06-19 that the example L1B values are not real; this discrepancy will be
  re-evaluated once real L1B data and MODTRAN 6 coefficients are available.

---

## 4. Hardcoded / Duplicated Constants

### Cloud fraction and DCC thresholds are duplicated

- `unfiltered_radiances/unfiltering.py:16–17`:
  `_CLOUD_FRACTION_THRESHOLD = 0.10`, `_DCC_OT_THRESHOLD = 10.0`
- `tp7/modtran6.py:30–31`:
  `DCC_CLDC_THRESHOLD = 0.10`, `DCC_CLD_OT_THRESHOLD = 10.0`
- Same values in two places. If thresholds are changed, both files must be updated.

### IGBP mapping only handles two surface types explicitly (`unfiltering.py:18–19`)

- IGBP 17 (Water Bodies) → Ocean; IGBP 15 (Snow and Ice) → Snow.
- All other 18 IGBP types (including permanent wetlands, tundra, urban) default to **Land**.
- Appropriate for now but may need refinement as more scene types are evaluated.

### Angle bin edges (`unfiltering.py:12–14`)

- Samples with SZA > 85°, VZA > 90°, or RAZ (after fold) outside [0°, 180°] get bin index −1
  and silently produce NaN output.
- No quality flag distinguishes this case from sparse-bin NaN or out-of-range scene NaN.

```python
_SZA_EDGES = [0.0, 22.2, 41.4, 60.0, 75.5, 85.0]
_VZA_EDGES = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
_RAZ_EDGES = [0.0, 15.0, 60.0, 120.0, 165.0, 180.0]
```

---

## 5. Outstanding TODOs (Code Debt)

| Location | Description |
|---|---|
| `tp7/modtran6.py:45, 51` | `load_srf()` and `get_interpolated_srf()` are duplicated from `tp7/tp7.py`. Should be a shared `srf_utils` module. |
| `prod/std/standard_method.py:107–108` | S3 path support via `AnyPath`/`cloudpathlib` is in place but has not been tested against a real S3 bucket. |
| `prod/std/standard_method.py:116` | No unified parser — `.tp7` and `.nc` inputs use separate `load_dataset()` / `load_nc_dataset()` functions. |
| `tests/unit/` | No unit tests for `apply_unfiltering()`, `classify_scene_cloud()`, or `_assign_bin_indices()` in `unfiltering.py`. |

---

## 6. Missing Features

### Standard vs. multivariate method switch

- No `method` parameter exists to choose between:
  - `"standard"` — univariate regression for all channels (traditional CERES approach)
  - `"multivariate"` — multivariate for SSW/SW, univariate for LW/TOT (current implementation)
- Referenced in `notes/next_steps_2026-06-12.md` as a required deliverable.

### Coefficient uncertainty estimates

- Coefficients are point estimates only. No fit residuals, R², or confidence intervals are stored
  in the coefficient NetCDF file.
- Needed for downstream uncertainty propagation in the L2 product.

### Per-sample quality flags for unfiltering failures

- `quality_flags` in the output currently copies the L1B `Quality_Flag` verbatim.
- No bits are set to distinguish between:
  - Valid unfiltered output
  - NaN due to sparse/missing coefficient bin
  - NaN due to out-of-range angles (SZA > 85°, VZA > 90°)
  - NaN due to unclassified/untrained scene type
