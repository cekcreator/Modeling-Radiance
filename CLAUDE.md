# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

There are two separate environments depending on which part of the repo you're working in.

**Research notebooks** use conda with the pinned lockfile at `research/notebooks/requirements.txt` (conda `--file` format, osx-arm64):

```bash
conda create --name mr_env --file research/notebooks/requirements.txt
conda activate mr_env
```

`env.yaml` is an alternate conda env file for the same environment.

**Production package** (`unfiltered_radiances/`, `prod/`) uses Poetry. `pyproject.toml` declares deps and uses `poetry-core` as the build backend. There is no committed `poetry.lock` yet — Poetry will generate one on first run:

```bash
poetry install
```

The Dockerfile runs `poetry lock && poetry sync --only main --no-root` at build time. The production package's importable namespaces are `tp7`, `srfs`, and `matt_code` per `[tool.setuptools.packages.find]` in `pyproject.toml`.

## Architecture Overview

There are two distinct areas of the repo with different purposes:

### Research (`research/`)
Experimental notebooks and supporting utilities. The primary notebook for the full end-to-end pipeline is `research/notebooks/nb_pt1.ipynb`. Other notebooks (`modeling.ipynb`, `create_dataset.ipynb`, etc.) are earlier-stage experiments. `research/matt_code/` contains alternate parsers and SRF helpers used by notebooks.

### Production (`prod/` and `unfiltered_radiances/`)
- `prod/std/standard_method.py` — generates quadratic regression coefficients per scene/cloud/SZA/VZA/RAZ bin and serializes them to a NetCDF cube. This is the active implementation of the traditional unfiltering method.
- `unfiltered_radiances/unfiltering.py` — core science logic. `apply_unfiltering()` takes filtered radiances + angles + the coefficient dataset and returns four unfiltered channels (SW, SSW, LW, TOT). `HARDCODED_SCENE` and `HARDCODED_CLOUD` are temporary constants at the top — they will eventually be replaced by per-sample lookup from the FMATCH-CAM ancillary file. Bin edges: SZA `[0, 22.2, 41.4, 60.0, 75.5, 85.0]`, VZA `[0, 15, 30, 45, 60, 90]`, RAZ `[0, 15, 60, 120, 165, 180]`.
- `unfiltered_radiances/algorithm.py` — Libera SDC processing entrypoint (reads a manifest, calls `calculate_science_data()`, writes a NetCDF product via `libera_utils`). Fully implemented — reads filtered radiances and angles from the L1B file, calls `apply_unfiltering()`, and returns all output variables including SZA/VZA/RAZ passthrough.
- `unfiltered_radiances/l2-unfiltered-radiance-product-definition.yml` — defines the output NetCDF schema (variables, units, dimensions). Variable names here must match keys returned by `calculate_science_data()`. `ProductID: UNF-RAD-CAM`.

The production algorithm runs as a Docker container. The Dockerfile is at the repo root. When building, `PYTHONPATH=/app` must be set so the local packages (`tp7`, `srfs`, `prod`, `unfiltered_radiances`) are importable without an editable install, and the ENTRYPOINT must point to `unfiltered_radiances/algorithm.py` (not a flat-copied `algorithm.py`). Required runtime env vars: `PROCESSING_PATH` (output directory); optional: `COEFFICIENTS_FILE` (override default coefficient file lookup).

### Core Library Packages

**`tp7/tp7.py` — `Tape7` class**
The central data-ingestion object. Instantiate with a `.tp7` filepath and it runs the full pipeline automatically:
1. `_read_tp7()` — parses the file into a 3D numpy grid `(wavelength_points, columns, runs)` plus raw header lines
2. `_build_scene_description()` — parses header lines into `describer_df`, a DataFrame with one row per run containing SZA, VZA, RAZ, scene metadata
3. `_compute_radiences()` — converts frequency-domain MODTRAN output to wavelength-domain; produces `rads` array shaped `(runs, 3, 4000)` where axis 1 is `[wavelength, SW_radiance, LW_radiance]`
4. `_integrate_radiances()` — applies Libera SRFs and integrates with Simpson's rule; appends six columns to `describer_df`: `Shortwave Unfiltered Rads (Integrated)`, `Longwave Unfiltered Rads (Integrated)`, `Shortwave Filtered Rads (Integrated)`, `Longwave Filtered Rads (Integrated)`, `Split Shortwave Filtered Rads (Integrated)`, `Total Filtered Rads (Integrated)`

`describer_df` is the primary output used by downstream modeling and regression code.

**`srfs/srfs.py` — `SRFS` class**
Loads an SRF CSV from `data/SRF/` and builds a range-keyed DataFrame. `Tape7` bypasses this class and calls `get_interpolated_srf()` directly, using `np.interp` to map SRF onto radiance wavelengths before integration.

SRF CSVs live at `data/SRF/libera_srf_{channel}_v0-0-1.csv` where channel is `sw`, `ssw`, `lw`, or `total`.

**`tp7/modtran6.py` — `load_nc_dataset(data_dir)`**
Parses MODTRAN 6 NetCDF files and returns a `describer_df`-compatible DataFrame with the same columns as `Tape7.describer_df`. Called by `standard_method.py::run()` when `modtran_version` starts with `"6"`. MODTRAN 6 example files are under `data/Modtran_6_data/`. Note: the real M6 production data lives on S3 in the AWS SAE — local files are small synthetic examples for pipeline testing only.

## Dev Environment Gotchas

**Poetry is not on Claude Code's PATH.** It's only sourced in the user's interactive shell (`.zshrc`). The `.venv` at the repo root is fully populated by Poetry. Always use `.venv/bin/python -m pytest` instead of `pytest`, and never call `poetry` from bash tools — it will fail with "command not found". To add a dependency, run `poetry add <pkg>` in your own terminal.

**Two separate environments — never mix them:**
- `research/notebooks/` → conda `mr_env`
- `prod/`, `unfiltered_radiances/`, `tp7/`, `srfs/` → Poetry `.venv/`

## Key Gotchas

**Header parsing is fragile.** Two `.tp7` file formats exist — 12-column and 14-column. `_parse_metadata()` detects which by checking the second-to-last line. Column index mappings differ: `[1,6,10,5,4]` for 12-col, `[0,7,13,5,9]` for 14-col. Do not change these without inspecting actual `.tp7` files first.

**VZA flip for Land and Deep Convective Cloud.** After parsing, `vza = 180 - vza` is applied to these two scene types in `_build_scene_description()`. Other scene types read VZA directly from a different header line position (`headerdata[i][11]`).

**Card 3 splitting for Land/DCC VZA.** Some runs have run-together values in `card3[2]` (e.g., `"-12.345678901.234"`). The parser manually splits at position 9 to recover VZA. This is a known MODTRAN 3.7 formatting artifact.

**Frequency → wavelength conversion.** MODTRAN outputs in wavenumber (cm⁻¹). The conversion is `lam = (1 / freq) * 1e4` (µm), and the radiance conversion is `rad_wv = freq² * rad_freq`. Arrays are reversed (`[::-1]`) to go from ascending frequency to ascending wavelength.

**Scene type is inferred from filename prefix.** The mapping is: `lnd` → Land, `ocecld` → Cloudy Ocean, `oceclr` → Clear Ocean, `sno` → Snow, `dc` → Deep Convective Cloud. The `.tp7` files are organized under `data/Modtran_Unfiltering_Tape7s_SZA{00,41,60,75,85}/`.

## Coefficient Output

Generated coefficient files live in `coefficients/` at the repo root (tracked by git). Files are named `unfiltering_coefficients_v{semver}_srf-{srf_version}_modtran-{modtran_version}.nc` — e.g. `unfiltering_coefficients_v0.1.0_srf-0-0-1_modtran-3.7.nc`. Regeneration is triggered whenever the SRFs change. Run via:

```python
from prod.std.standard_method import run
run(data_dir="data/Modtran_3-7_data/", srf_dir="data/SRF/", srf_version="0-0-1", modtran_version="3.7")
```

## Standard Method (Regression Coefficients)

`prod/std/standard_method.py::run()` is the top-level entrypoint. It loads all MODTRAN data (dispatching to `Tape7` for MODTRAN 3.7 or `load_nc_dataset()` for MODTRAN 6), merges scenes, fits coefficients, and writes the output NetCDF.

`serialize_coefficients()` writes an xarray Dataset with a 5D coefficient cube shaped `(scene, cloud, sza_bin, vza_bin, raz_bin)`:
- `sw_coefficients` / `ssw_coefficients` — 7 params each (multivariate quadratic over SW + SSW filtered radiances, via `PolynomialFeatures(degree=2)`)
- `lw_coefficients` / `tot_coefficients` — 3 params each (univariate degree-2 polynomial)

Scene dimension: `["Land", "Cloudy Ocean", "Clear Ocean", "Snow", "Deep Convective Cloud"]`. Cloud dimension: `[0, 1]` (clear/cloudy). Bins that have no data are stored as NaN and skipped silently at inference time in `unfiltering.py`.

The `SCENE_TYPES` and `CLOUD_VALUES` lists defined in this file are the authoritative source — both `algorithm.py` and `unfiltering.py` import them directly.
