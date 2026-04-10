# AGENTS.md

This is the quick catch-up file for humans and coding agents working in this repo.

Update it whenever a major project decision changes.

## Project snapshot

- Project: IntelliPath
- Course: COMP 542 Machine Learning with AI
- Goal: build a traffic prediction / route assistance ML project that is realistic enough for a course demo but still manageable for a student team

## Current dataset decision

- Primary dataset candidate: LargeST GLA 2019
- Fallback / comparison dataset: METR-LA
- Why the team is leaning toward LargeST-GLA:
  - one full year of data in 2019
  - much larger LA-area graph
  - METR-LA appears to be mostly embedded inside the GLA subset

## Current known data facts

- METR-LA local file:
  - `data/raw/METR-LA-Complete/metr-la.h5`
  - shape: `34272 x 207`
  - time range: `2012-03-01 00:00:00` to `2012-06-27 23:55:00`

- LargeST-GLA local generated files:
  - `data/raw/LargeST/gla/gla_his_2019.h5`
  - `data/raw/LargeST/gla/gla_meta.csv`
  - `data/raw/LargeST/gla/gla_rn_adj.npy`
  - flow shape: `35040 x 3834`
  - time range: `2019-01-01 00:00:00` to `2019-12-31 23:45:00`
  - adjacency shape: `3834 x 3834`

## Important repo rules

- Do not commit LargeST raw or generated data to git.
- LargeST data stays local under `data/raw/LargeST/`.
- Use `LARGEST_GLA_SETUP.md` for download / generation instructions.
- Commit notebooks, docs, loaders, and small metadata only.

## Current useful notebooks

- `notebooks/01_metr_la_exploration.ipynb`
  - first-pass METR-LA exploration

- `notebooks/02_largest_gla_feasibility.ipynb`
  - checks whether LargeST-GLA is present and summarizes feasibility

- `notebooks/03_largest_gla_exploration.ipynb`
  - main exploratory notebook for LargeST-GLA 2019
  - includes metadata inspection, graph summary, temporal patterns, and geographic comparison with METR-LA

## Recommended next engineering steps

1. Create a small reusable loader in `src/` for:
   - `gla_his_2019.h5`
   - `gla_meta.csv`
   - `gla_rn_adj.npy`
2. Define the first prediction target clearly.
   - recommended starting point: short-horizon traffic flow forecasting, `15 minutes ahead`
3. Define a chronological train / validation / test split within 2019.
4. Implement a persistence baseline first.
5. Add one simple non-deep-learning baseline before larger graph models.

## Open questions

- Should the project focus only on forecasting, or still keep a route-assistance framing in the presentation?
- Should the first experiments use all 3834 nodes or start with a smaller subgraph for speed?
- Which evaluation horizon should be primary: 15 minutes, 30 minutes, or 60 minutes ahead?

## Good first actions for any agent

1. Read `AGENTS.md`, `PROJECT_CONTEXT.md`, and `TASKS.md`.
2. Check whether `data/raw/LargeST/gla/` exists locally.
3. Review `notebooks/03_largest_gla_exploration.ipynb`.
4. Avoid changing unrelated files or committing large local data.
