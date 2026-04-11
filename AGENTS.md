# AGENTS.md

Agent-only workflow notes for this repo.

Humans should start with `TEAM_GUIDE.md`.

Agents should read `TEAM_GUIDE.md` first, then this file.

## Source Of Truth

- `TEAM_GUIDE.md` is the main human-readable project status file.
- `TASKS.md` is the current working checklist.
- `PROJECT_CONTEXT.md` is the short planning summary.
- If you change project direction, baseline workflow, or recommended next steps, update `TEAM_GUIDE.md` in the same change.

## Current Working Assumptions

- Main dataset: `LargeST-GLA 2019`
- Fallback dataset: `METR-LA`
- Current starter task: traffic flow forecasting at `15 minutes ahead`
- Main exploration notebook: `notebooks/03_largest_gla_exploration.ipynb`
- Main repeatable experiment: `src/run_baseline_experiment.py`

## Repo Safety Rules

- Do not commit LargeST raw or generated data.
- LargeST data stays local under `data/raw/LargeST/`.
- Use `LARGEST_GLA_SETUP.md` for dataset setup instructions.
- Avoid changing unrelated files.
- Keep docs concise and synchronized when making project-level decisions.

## Good First Actions For Any Agent

1. Read `TEAM_GUIDE.md`.
2. Read `TASKS.md`.
3. Confirm whether `data/raw/LargeST/gla/` exists locally.
4. Review `notebooks/03_largest_gla_exploration.ipynb` before proposing dataset changes.
5. Review `src/run_baseline_experiment.py` before changing the forecasting baseline flow.

## When Updating The Repo

- Put reusable logic in `src/`, not only in notebooks.
- Keep human-facing explanations in `TEAM_GUIDE.md`.
- Keep agent-specific guardrails in `AGENTS.md`.
- Prefer updating existing docs over creating new overlapping docs.
