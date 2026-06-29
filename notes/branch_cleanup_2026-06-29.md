# Branch Cleanup Plan — 2026-06-29

## Current branch state (commits ahead of `master`)

| Branch | Commits ahead | What's unique |
|---|---|---|
| `generate-coefficient-cube` | 5 | M6 parser, coefficient generation, Matt's naming/metadata updates, repo cleanup — already merged `ml-m6` in |
| `ml-m6` | 4 | Same as above + `CLAUDE.md` update (commit `89a48ad`, 2026-06-29) |
| `docker-img-build` | 2 | Full pipeline wired up + Docker/geometry variables (committed: `2bcf854`, `8551860`) |
| `data-file-integration` | 1 | Full pipeline only (`2bcf854`) — Docker work is on `docker-img-build`, not here |
| `fix-units` | 1 | Same coeff-gen commit as `ml-m6`/`generate-coefficient-cube`, nothing unique |
| `rf-srf` | 0 | Nothing — fully merged into master already |

`generate-coefficient-cube` is the deepest/most up-to-date branch for the core algorithm. The Docker + geometry-variable work was committed separately on `docker-img-build` and was never merged anywhere.

## Merge order for the cleanup branch

1. Branch off `generate-coefficient-cube` (deepest base, has M6 + Matt's naming work)
2. Merge in `docker-img-build` (picks up full pipeline + Docker/geometry variables)
3. Cherry-pick or merge `ml-m6`'s `CLAUDE.md` commit (`89a48ad`) — the only thing `ml-m6` has that `generate-coefficient-cube` doesn't
4. Drop `fix-units`, `data-file-integration`, `rf-srf` — fully subsumed by the above, safe to delete after the merge branch is verified

## Known gaps to verify after merging (see `known_discrepancies_2026-06-22.md` for full detail)

- Dockerfile still has the broken flat-COPY / missing `WORKDIR` / `PYTHONPATH` issues noted in `CLAUDE.md` — confirm `docker-img-build`'s version actually fixed this before relying on it
- `.dockerignore` and `scripts/push_to_ecr.sh` existence should be re-checked post-merge
- `CERES_SCENE_MAP` in `tp7/modtran6.py` only covers 2 of 5 scenes — merging branches won't fix this, still blocked on real M6 data from the SAE