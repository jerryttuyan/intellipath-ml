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

Current baseline models compared by default: `Persistence`, `Linear Regression`, and `Random Forest`.

## Reproducible Overrides (CLI)

The baseline runner accepts optional CLI flags so teammates and instructors can reproduce runs without editing code.

```bash
python src/run_baseline_experiment.py --help
```

Examples:

```bash
# Evaluate first 100 nodes using all CPU cores
python src/run_baseline_experiment.py --num-target-nodes 100 --rf-n-jobs -1 --show-config

# Evaluate one specific sensor with smaller forest for quick debugging
python src/run_baseline_experiment.py --target-node 767494 --rf-n-estimators 20 --rf-n-jobs 1 --show-config
```

Shared non-device presets are stored in `config/presets/*.json`:

```bash
# View available shared presets
python src/run_baseline_experiment.py --list-presets

# Run with a shared preset, then override local hardware settings
python src/run_baseline_experiment.py --preset report_baseline --rf-n-jobs -1 --show-config

# Save current non-device settings as a new shared preset
python src/run_baseline_experiment.py --preset default --num-target-nodes 300 --save-preset report_300_nodes
```

Shared multi-run suites are stored in `config/suites/*.json`:

```bash
# Run the required evaluation suite in one command
python src/run_experiment_suite.py --suite required_suite
```

`required_suite` includes:

- forecasting matrix runs (multiple horizons/slices)
- one routing comparison run (Shortest Distance vs Current Traffic vs Predicted Traffic)

Each run now also saves history artifacts automatically:

- `results/runs/<timestamp>_comparison.csv`
- `results/runs/<timestamp>_summary.csv`
- `results/run_history.csv` (append-only metadata + metrics)

Disable history for a one-off run with:

```bash
python src/run_baseline_experiment.py --no-save-history
```

## Optional UI (Streamlit)

For demos or non-coding teammates, launch the baseline runner UI:

```bash
streamlit run src/ui/baseline_ui.py
```

The UI exposes the same settings as the CLI, supports loading/saving shared presets, and runs the same baseline pipeline.

### Streamlit Workflow

1. Pick a preset from the sidebar and click `Apply preset`.
2. Adjust run-specific settings.
3. Click `Run Baseline Experiment`.
4. Monitor `Live Run Output` and status (`running`, `completed`, `failed`, `stopped`).
5. Use `Stop Run` if you need to terminate a long experiment.

To run required experiments automatically:

- select a suite in the sidebar under `Experiment Suites`
- click `Run Selected Suite`

What the UI shows:

- live tailed console output while the run is active
- current elapsed run time and last run duration
- summary and per-node result tables
- run-history trend charts (MAE/RMSE/runtime)

Where UI logs are stored:

- `results/ui_logs/*.log`

## Routing Experiment

Run a repeatable routing comparison (Shortest Distance vs Current Traffic vs Predicted Traffic):

```bash
python src/run_routing_experiment.py --prediction-model random_forest
```

Key outputs:

- `results/routing_comparison.csv` (per-route outcomes)
- `results/routing_summary.csv` (aggregated method comparison)
- `results/routing_runs/<timestamp>_*` (snapshots)
- `results/routing_history.csv` (append-only run history)

## Repo Map

- `notebooks/` exploratory work and visuals
- `src/` repeatable loaders, features, models, evaluation, runners
- `data/raw/` local datasets
- `results/` generated experiment outputs

## Notes

- Do not commit LargeST raw or generated data.
- Keep `TEAM_GUIDE.md` updated when the project direction changes.
