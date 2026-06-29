# Next Steps — 2026-06-12

## 1. Fix the unit mismatch (blocker for scientific accuracy)

**Problem:** The regression was trained on MODTRAN 3.7 radiances in `W sr⁻¹ cm⁻²` (integrated
over µm). The L1B `Filtered_Radiance_*` variables are labeled `W/(m²·sr·nm)` and have values
~100× larger (SW: ~3200 vs training SW: ~4–481). This mismatch makes the current output
unfiltered radiances physically meaningless.

**Investigation needed:**
- Confirm the true units and physical meaning of `Filtered_Radiance_SW/LW/SSW/Tot` in the L1B.
  The label `W/(m²·sr·nm)` is suspicious for a broadband filtered quantity — it may already be
  SRF-integrated but mislabeled.
- Confirm the exact units that `Tape7._integrate_radiances()` produces (integration over µm gives
  `W sr⁻¹ cm⁻²`).
- Either: apply a conversion factor before passing L1B radiances into `apply_unfiltering()`, or
  regenerate coefficients using MODTRAN 6 data with units that match the L1B scale (preferred —
  see item 2).

**Where to fix:** `unfiltered_radiances/unfiltering.py::apply_unfiltering()` — add unit conversion
of input radiances before the polynomial evaluation, or document the expected input units and
enforce them upstream.

---

## 2. Generate MODTRAN 6 coefficients from AWS

**Status:** `Modtran6NC` parser and `load_nc_dataset()` are implemented and tested. AWS path
support in `load_nc_dataset()` still needs to be verified (it uses `AnyPath` / `cloudpathlib`
which should handle S3 URLs, but hasn't been exercised against a real bucket).

**Steps:**
1. Get the S3 bucket path / prefix for the MODTRAN 6 `.nc` files from the team.
2. Run `load_nc_dataset()` against the S3 prefix and confirm it loads without errors.
3. Expand `CERES_SCENE_MAP` in `tp7/modtran6.py` as new scene IDs appear in the M6 files
   (currently only 5=Clear Ocean and 18=Cloudy Ocean are mapped).
4. Run `generate_multivariate_coefficients()` on the combined M6 dataset.
5. Save the output to `coefficients/` with a new semver and `modtran-6.x` in the filename (e.g.
   `unfiltering_coefficients_v0.2.0_srf-0-0-1_modtran-6.0.nc`).
6. Re-run the algorithm end-to-end against the L1B example file and verify the output radiance
   ranges are physically plausible.

**Key code:**
```python
from prod.std.standard_method import run
run(data_dir="s3://bucket/prefix/", srf_dir="data/SRF/", srf_version="0-0-1", modtran_version="6.0")
```

---

## 3. Wire in the scene/cloud classification input file

**Status:** `algorithm.py` currently hardcodes `HARDCODED_SCENE = "Clear Ocean"` and
`HARDCODED_CLOUD = 0`, producing coefficients for only one bin and leaving 438k/602k samples as
NaN.

**When the scene/cloud file arrives:**
1. Identify the file format and which variable maps to CERES scene ID and cloud fraction.
2. Add it to the manifest (new `ProductID` — TBD with team).
3. In `_get_scene_cloud_dataset()` (to be added to `algorithm.py`), load and index by
   `radiometer_time` or sample index.
4. Remove `HARDCODED_SCENE` / `HARDCODED_CLOUD` from `unfiltering.py` and replace
   `apply_unfiltering()` call with a vectorized per-sample scene/cloud lookup.
5. Map scene IDs in the classification file to the 5 scene type strings used in the coefficient
   file (same mapping as `CERES_SCENE_MAP` in `modtran6.py`).

---

## 4. Add a standard-method / multivariate switch

**From SDC notes (2026-06-04):** we need "a switch to flip between standard method and
multivariate."

**Current state:** SW/SSW use multivariate quadratic regression (2 inputs → 7 coefs). LW/TOT
use univariate quadratic (1 input → 3 coefs). The distinction is baked in but not user-exposed.

**Design:** Add a `method` parameter (e.g. `"standard"` vs `"multivariate"`) to `run()` in
`standard_method.py` and to `apply_unfiltering()` in `unfiltering.py`. The `"standard"` path
would use only SW → SW unfiltered (univariate) for all channels, matching the traditional CERES
approach. Keep both coefficient sets in the same `.nc` file or in separate files with a clear
naming convention.

---

## 5. Scientific validation (end-of-June deadline)

**Goal:** Produce a realistic unfiltered data product that can be scientifically validated by mid-
to-late June.

**Steps:**
1. Resolve unit mismatch (item 1) and regenerate M6 coefficients (item 2).
2. Run the algorithm on the example L1B file with full scene/cloud coverage.
3. Compare unfiltered radiance to pre-integrated MODTRAN 6 `MODTRAN6_SW_RAD_TOA_CERES_TRMM`
   values as a ground-truth reference — the regression output should closely match.
4. Check the LW/SW ratio and per-scene statistics for physical plausibility.
5. Share a sample output file with whoever needs a realistic unfiltered product by end of June.

---

## 6. Minor cleanup items

- **`CERES_SCENE_MAP` completeness:** Land, Snow, and DCC (via regular scene ID) are not yet in
  the map. Extend once M6 files with those scenes are available.
- **`valid_range` in product YAML:** Currently set to `[0, 1000]` for all radiance channels.
  Update after the unit mismatch is resolved and real value ranges are known.
- **Tests for `unfiltering.py`:** No unit tests yet for `apply_unfiltering()` or the bin-index
  helpers. Add to `tests/unit/` alongside the existing `test_modtran6.py`.
- **`ssw_regression_method` attribute:** Currently states "Multivariate quadratic regression" in
  the YAML. Confirm this is the correct label once the method switch (item 4) is designed.
