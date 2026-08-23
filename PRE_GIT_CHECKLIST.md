# Pre-Git Checklist

Run through this BEFORE any `git add` / `git push` on this directory.

## Credential Safety

- [ ] No WRDS password in any file (`grep -rn "password\|pgpass\|wrds-pgdata" markov_vol/`)
- [ ] No username hardcoded in markov_vol/ (`grep -rn "sjmagill" markov_vol/`)
- [x] `fetch_daily_trace.py` has a self-contained `_connect()` reading `WRDS_USERNAME` env + pgpass -- no username or password in source
- [x] All cross-repo impcredvol/impcredvol2 paths replaced with env-var overrides (`MONTHLY_PORTFOLIOS_PATH`, `CRSP_DAILY_PATH`, `CIV_RAW_DIR`, `CIV_INT_DIR`)

## Large Files

- [ ] `.gitignore` excludes all parquet files (`data/*.parquet`)
- [ ] `.gitignore` excludes results (`data/results/*.json`, `data/results/*.npz`)
- [ ] No file over 1 MB is staged (`git diff --cached --stat` -- check sizes)
- [ ] `daily_trace.parquet` (927 MB) is NOT staged

## Stale Artifacts

- [ ] No `__pycache__/` directories staged
- [ ] No `.pyc` files staged
- [ ] No LaTeX build artifacts (`.aux`, `.log`, `.out`, `.toc`)
- [ ] No leftover results from previous runs in `data/results/`

## Content Review

- [ ] README.md has no personal identifiers beyond author name
- [ ] PITFALLS.md has no credential information
- [ ] All print statements are ASCII (no Unicode -- Windows console issue)
- [ ] No absolute paths hardcoded (all paths are relative to PROJECT_ROOT)

## Before Push

```bash
# Run these checks
grep -rn "sjmagill\|password\|pgpass.conf" markov_vol/ --include="*.py" --include="*.md"
git diff --cached --stat | grep -E "\+.*MB"
find markov_vol/ -name "__pycache__" -o -name "*.pyc"
```

If any of these produce output, fix before pushing.
