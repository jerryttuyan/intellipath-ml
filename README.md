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

## Environment

Everyone should use an isolated Python environment for this repo.

`venv` is the simplest shared default, but the exact environment name does not matter as long as:

- you install the repo requirements into it
- you use that same environment for scripts and notebooks

Example setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Run

```bash
python src/run_baseline_experiment.py
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
