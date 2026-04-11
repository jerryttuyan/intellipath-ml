# Team Guide

Read this first if you want to get up to speed quickly.

## Project At A Glance

| Item | Current answer |
|---|---|
| Project | IntelliPath |
| Course | COMP 542 Machine Learning with AI |
| Main goal | traffic forecasting first, route-assistance framing second |
| Primary dataset | `LargeST-GLA 2019` |
| Fallback dataset | `METR-LA` |
| Main notebook | `notebooks/03_largest_gla_exploration.ipynb` |
| Main experiment script | `src/run_baseline_experiment.py` |

## What We Decided

| Topic | Decision |
|---|---|
| Dataset | use `LargeST-GLA 2019` as the main working dataset |
| Why | it gives us a full 2019 year, many more sensors, and still stays LA-relevant |
| Forecast target | short-horizon traffic flow forecasting |
| Starting horizon | `15 minutes ahead` |
| Current baseline comparison | `Persistence` vs `Linear Regression` vs `Random Forest` |

## Dataset Comparison

| Dataset | Time coverage | Sensors | Role for us |
|---|---|---:|---|
| `METR-LA` | Mar 1, 2012 to Jun 27, 2012 | 207 | fallback and comparison dataset |
| `LargeST-GLA 2019` | Jan 1, 2019 to Dec 31, 2019 | 3834 | main dataset for experiments |

## What To Open

| If you want to... | Open this |
|---|---|
| understand the whole project fast | `TEAM_GUIDE.md` |
| see the repo landing page | `README.md` |
| see agent workflow / repo rules | `AGENTS.md` |
| see current plan and direction | `PROJECT_CONTEXT.md` |
| see the active checklist | `TASKS.md` |
| reproduce the dataset setup | `LARGEST_GLA_SETUP.md` |

## What To Run

| Goal | Run / open | What you should get |
|---|---|---|
| explore the main dataset | `notebooks/03_largest_gla_exploration.ipynb` | dataset shape, time range, metadata, adjacency, maps, temporal plots |
| run the forecasting baseline | `python src/run_baseline_experiment.py` from your activated project environment | baseline metrics in terminal and CSV outputs |
| run baseline with reproducible overrides | `python src/run_baseline_experiment.py --num-target-nodes 100 --rf-n-jobs -1 --show-config` | same pipeline with explicit runtime settings printed in terminal |
| run baseline with a shared preset | `python src/run_baseline_experiment.py --preset report_baseline --rf-n-jobs -1 --show-config` | portable team settings plus local hardware override |
| run required evaluation suite (one command) | `python src/run_experiment_suite.py --suite required_suite` | executes required forecasting matrix + routing comparison sequentially |
| run baseline with UI | `streamlit run src/ui/baseline_ui.py` | browser-based controls for settings, run logs, and result tables/charts |
| inspect summarized baseline results | `results/baseline_summary.csv` | average performance by model |
| inspect run history across experiments | `results/run_history.csv` and `results/runs/` | timestamped artifacts and an append-only record for plotting trends |
| run routing comparison experiment | `python src/run_routing_experiment.py --prediction-model random_forest` | shortest-distance vs current-traffic vs predicted-traffic route outcomes |

UI runtime notes:

- click `Apply preset` after selecting a preset in the sidebar
- the UI supports `Run` and `Stop` controls for long jobs
- live run output is shown as a tail view (latest lines)
- status and timing are shown in-page (`running/completed/failed/stopped`, current elapsed, last run duration)

## Baseline Runtime Settings

Random Forest baseline knobs live in `src/config.py`:

- `rf_n_estimators`: number of trees (higher can improve accuracy but runs slower)
- `rf_random_state`: random seed for reproducible comparisons across teammates
- `rf_n_jobs`: CPU cores used by Random Forest (`1` single core, `-1` all cores, or a positive cap like `2`/`4`)

Recommended team convention:

- For shared/fair result comparison: keep `rf_random_state=42` and use a consistent `rf_n_jobs` setting
- For faster local iteration: set `rf_n_jobs=-1` if your machine is idle
- If your laptop is lagging: reduce `rf_n_jobs` to `1` or `2`

Shared preset files (non-device settings only):

- `config/presets/default.json`
- `config/presets/quick_debug.json`
- `config/presets/report_baseline.json`

Shared suite files (multi-run automation):

- `config/suites/required_suite.json`

## Main Files

| Path | Purpose |
|---|---|
| `notebooks/01_metr_la_exploration.ipynb` | first-pass METR-LA exploration |
| `notebooks/02_largest_gla_feasibility.ipynb` | check whether LargeST-GLA is locally available and usable |
| `notebooks/03_largest_gla_exploration.ipynb` | main LargeST-GLA EDA notebook |
| `src/data/gla_loader.py` | load GLA data and make chronological splits |
| `src/features/baseline_features.py` | build simple lag and time features |
| `src/models/persistence.py` | persistence baseline |
| `src/models/linear_regression_baseline.py` | linear regression baseline |
| `src/models/random_forest_baseline.py` | random forest baseline |
| `src/evaluation/metrics.py` | evaluation metrics |
| `src/run_baseline_experiment.py` | repeatable baseline runner |
| `src/run_routing_experiment.py` | repeatable routing comparison runner |

## Current Baseline Snapshot

Latest local summary across 10 GLA sensors:

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| Random Forest | 12.3554 | 17.9643 |
| Linear Regression | 14.2963 | 20.1362 |
| Persistence | 15.4667 | 21.7013 |

Plain-English read:

- Random Forest is the strongest baseline among the current three models.
- Linear Regression beats Persistence, but trails Random Forest.
- That is a solid first result, but it is still an early baseline, not the final project answer.

## How The Repo Works Right Now

1. Use notebooks to understand the data and create presentation-friendly visuals.
2. Move reusable logic into `src/`.
3. Run repeatable experiments from scripts.
4. Save generated metrics to `results/`.
5. Use notebook plots and result summaries for interpretation and slides.

## Data Layout

| Path | What belongs there |
|---|---|
| `data/raw/METR-LA-Complete/` | local METR-LA files |
| `data/raw/LargeST/ca/` | local LargeST California source files |
| `data/raw/LargeST/gla/` | generated GLA subset files |
| `results/` | generated experiment outputs |

Expected main GLA files:

- `data/raw/LargeST/gla/gla_his_2019.h5`
- `data/raw/LargeST/gla/gla_meta.csv`
- `data/raw/LargeST/gla/gla_rn_adj.npy`

## What Not To Commit

- anything under `data/raw/LargeST/`
- machine noise like `.DS_Store`
- temporary local caches or checkpoint files

## Best Next Steps

1. Decide how much of the final presentation should be forecasting versus routing.
2. Expand the baseline run beyond the current 10-node sample.
3. Add clearer result plots for model comparison.
4. Decide whether the next model stays tabular or moves toward graph-based forecasting.

## Keeping Everyone In Sync

- Humans should read this file first.
- Agents should read this file first, then `AGENTS.md`.
- If the project direction changes, update this file in the same change.
