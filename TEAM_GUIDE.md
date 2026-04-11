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
| Current baseline comparison | `Persistence` vs `Random Forest` |

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
| run the forecasting baseline | `./venv/bin/python src/run_baseline_experiment.py` | baseline metrics in terminal and CSV outputs |
| inspect summarized baseline results | `results/baseline_summary.csv` | average performance by model |

## Main Files

| Path | Purpose |
|---|---|
| `notebooks/01_metr_la_exploration.ipynb` | first-pass METR-LA exploration |
| `notebooks/02_largest_gla_feasibility.ipynb` | check whether LargeST-GLA is locally available and usable |
| `notebooks/03_largest_gla_exploration.ipynb` | main LargeST-GLA EDA notebook |
| `src/data/gla_loader.py` | load GLA data and make chronological splits |
| `src/features/baseline_features.py` | build simple lag and time features |
| `src/models/persistence.py` | persistence baseline |
| `src/models/random_forest_baseline.py` | random forest baseline |
| `src/evaluation/metrics.py` | evaluation metrics |
| `src/run_baseline_experiment.py` | repeatable baseline runner |

## Current Baseline Snapshot

Latest local summary across 10 GLA sensors:

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| Random Forest | 12.3554 | 17.9643 |
| Persistence | 15.4667 | 21.7013 |

Plain-English read:

- Random Forest is beating the simple persistence baseline.
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
