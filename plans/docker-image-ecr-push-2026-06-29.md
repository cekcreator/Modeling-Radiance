# Plan: Fix Dockerfile + ECR Push Script
## 2026-06-29

## Context

The original Dockerfile copied individual .py files to `/` with no WORKDIR, breaking all
relative import paths. This rewrites it to copy the full package tree and sets PYTHONPATH.

## Changes Made

### `Dockerfile` — full rewrite
- Added `WORKDIR /app`
- Copies all source packages: `unfiltered_radiances/`, `prod/`, `tp7/`, `srfs/`
- Copies `data/SRF/` (needed by tp7.py's `_DEFAULT_SRF_DIR`)
- Copies `coefficients/` (auto-discovered at runtime by `_find_coefficient_file()`)
- Sets `ENV PYTHONPATH=/app` so local packages are importable without editable install
- ENTRYPOINT is `python unfiltered_radiances/algorithm.py` (not broken root path)

### `.dockerignore` — new
- Excludes `.git/`, `.venv*/`, `data/` (large MODTRAN files), `research/`, `tests/`
- Re-includes `data/SRF/` via `!data/SRF/` negation

### `scripts/push_to_ecr.sh` — new
- Parameterized by `AWS_ACCOUNT_ID`, `AWS_REGION`, `IMAGE_NAME`, `IMAGE_TAG`
- Builds, authenticates with ECR, tags, and pushes

### `COMMANDS.txt` — added Docker section
- `docker build` command
- `docker run` test with volume mounts
- ECR push instructions
