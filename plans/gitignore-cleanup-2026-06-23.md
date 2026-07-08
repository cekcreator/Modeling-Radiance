# Plan: .gitignore Cleanup + Repo Plans Directory
## 2026-06-23

## Context

The `.gitignore` had grown out of sync with repo intent:
- `CLAUDE.md` and `notes/` were added to gitignore by mistake — they are project context
  documents that should be tracked.
- Generated artifacts (`coefficients/*.nc`, `unfiltering.egg-info/`) were tracked in git,
  causing merge conflicts when coefficient files are regenerated.
- Common generated paths (`*.log`, `.pytest_cache/`) were missing from the ignore list.

Also established `plans/` as the tracked directory for all future plan documents, named
`plans/<title>-YYYY-MM-DD.md`.

## Changes Made

### `.gitignore`
- Removed `CLAUDE.md` and `notes/` (now tracked)
- Added `coefficients/*.nc`, `unfiltering.egg-info/`, `*.log`, `.pytest_cache/`

### Untracked
- `coefficients/unfiltering_coefficients_v0.1.0_srf-0-0-1_modtran-3.7.nc`
- `unfiltering.egg-info/` (5 files)

### Newly Tracked
- `CLAUDE.md`
- `notes/known_discrepancies_2026-06-22.md`
- `plans/` (this directory)
