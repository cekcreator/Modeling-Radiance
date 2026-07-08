# Plan: Coefficient Metadata Improvements + FMATCH Rename
## 2026-06-29

## Context

From `notes/matt_sdc_meeting_2026-06-25.md`, Matt requested:
1. Longer scene descriptions in the coefficient NetCDF (indexes → IGBP/cloud rules)
2. Explicit bin range inclusivity documentation
3. Rename ancillary file references from "CAM" to "FMATCH"; update ProductID to `UNF-RAD-CAM`

## Changes Made

### `prod/std/standard_method.py` — `serialize_coefficients()`
- Added `description` attribute to `scene` coordinate with IGBP/cloud rule for each index (0–4)
- Updated `sza_lo`, `sza_hi`, `vza_lo`, `vza_hi`, `raz_lo`, `raz_hi` long_names to say "inclusive"
- Added global attribute `bin_convention: "All angle bins are inclusive on both lower and upper bounds."`

### `unfiltered_radiances/l2-unfiltered-radiance-product-definition.yml`
- Updated `ProductID` from `UNF-RAD` → `UNF-RAD-CAM`