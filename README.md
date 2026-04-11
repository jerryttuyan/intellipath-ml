# IntelliPath ML

Traffic forecasting project for COMP 542.

## Start Here

- Humans: `TEAM_GUIDE.md`
- Agents: `TEAM_GUIDE.md`, then `AGENTS.md`
- LargeST setup: `LARGEST_GLA_SETUP.md`

## Current Direction

- Main dataset: `LargeST-GLA 2019`
- Fallback dataset: `METR-LA`
- Current baseline path: `src/run_baseline_experiment.py`
- Main exploration notebook: `notebooks/03_largest_gla_exploration.ipynb`

## Quick Run

```bash
./venv/bin/python src/run_baseline_experiment.py
```

This writes baseline metrics to:

- `results/baseline_comparison.csv`
- `results/baseline_summary.csv`

## Repo Map

- `notebooks/` exploratory work and visuals
- `src/` repeatable loaders, features, models, evaluation, runners
- `data/raw/` local datasets
- `results/` generated experiment outputs

## Notes

- Do not commit LargeST raw or generated data.
- Keep `TEAM_GUIDE.md` updated when the project direction changes.
