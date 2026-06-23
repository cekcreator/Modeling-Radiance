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
- `prod/std/standard_method.py` — generates quadratic regression coefficients per SZA/VZA/RAZ/scene bin. This is the active implementation of the traditional unfiltering method.
- `unfiltered_radiances/algorithm.py` — Libera SDC processing template (reads a manifest, calls science logic, writes a NetCDF product via `libera_utils`). The `calculate_science_data()` function is currently a placeholder and is where the regression unfiltering algorithm will be wired in.
- `unfiltered_radiances/l2-unfiltered-radiance-product-definition.yml` — defines the output NetCDF schema (variables, units, dimensions). Variable names here must match keys returned by `calculate_science_data()`.

The production algorithm runs as a Docker container: `ENTRYPOINT ["python", "algorithm.py"]` with a manifest path as the CLI argument.

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

## Key Gotchas

**Header parsing is fragile.** Two `.tp7` file formats exist — 12-column and 14-column. `_parse_metadata()` detects which by checking the second-to-last line. Column index mappings differ: `[1,6,10,5,4]` for 12-col, `[0,7,13,5,9]` for 14-col. Do not change these without inspecting actual `.tp7` files first.

**VZA flip for Land and Deep Convective Cloud.** After parsing, `vza = 180 - vza` is applied to these two scene types in `_build_scene_description()`. Other scene types read VZA directly from a different header line position (`headerdata[i][11]`).

**Card 3 splitting for Land/DCC VZA.** Some runs have run-together values in `card3[2]` (e.g., `"-12.345678901.234"`). The parser manually splits at position 9 to recover VZA. This is a known MODTRAN 3.7 formatting artifact.

**Frequency → wavelength conversion.** MODTRAN outputs in wavenumber (cm⁻¹). The conversion is `lam = (1 / freq) * 1e4` (µm), and the radiance conversion is `rad_wv = freq² * rad_freq`. Arrays are reversed (`[::-1]`) to go from ascending frequency to ascending wavelength.

**Scene type is inferred from filename prefix.** The mapping is: `lnd` → Land, `ocecld` → Cloudy Ocean, `oceclr` → Clear Ocean, `sno` → Snow, `dc` → Deep Convective Cloud. The `.tp7` files are organized under `data/Modtran_Unfiltering_Tape7s_SZA{00,41,60,75,85}/`.

## Coefficient Output

Generated coefficient files live in `coefficients/` at the repo root (tracked by git). Files are named `unfiltering_coefficients_srf-{srf_version}_{timestamp}.nc` so each file is traceable to the SRF version that produced it. Regeneration is triggered whenever the SRFs change. Run via:

```python
from prod.std.standard_method import run
run(data_dir="data/Modtran_3-7_data/", srf_dir="data/SRF/", srf_version="0-0-1", modtran_version="3.7")
```

## Standard Method (Regression Coefficients)

`prod/std/standard_method.py::generate_multivariate_coefficients(dataset)` takes a combined `describer_df` DataFrame (all scenes merged) and returns a dict keyed by `(sza_bin, vza_bin, raz_bin)` tuples. Each value is a tuple `(sw_coef, ssw_coef, lw_coef)`:
- `sw_coef` / `lw_coef` — degree-2 polynomial coefficients from `np.polynomial.polynomial.Polynomial.fit`
- `ssw_coef` — multivariate regression coefficients using both SSW and SW filtered radiances as inputs (`sklearn.linear_model.LinearRegression` on `PolynomialFeatures(degree=2)` expansion)

Bins that have no data (sparse angular combinations) are silently skipped with a print statement.
