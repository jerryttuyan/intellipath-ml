# Project Context

Short planning summary. For the quick human-readable catch-up, use `TEAM_GUIDE.md`.

## Goal

Build a manageable course project around traffic forecasting, with route assistance kept as a presentation framing if useful.

## Current Direction

- Main dataset: `LargeST-GLA 2019`
- Fallback dataset: `METR-LA`
- Main task: short-horizon traffic flow forecasting
- Starting horizon: `15 minutes ahead`

## Current Plan

1. Keep `LargeST-GLA` as the primary experiment dataset.
2. Use notebooks for exploration and presentation visuals.
3. Keep reusable experiment logic in `src/`.
4. Compare a simple persistence baseline against stronger tabular baselines first.
5. Decide later whether to move into graph-based forecasting models.

## Constraints

- student project scope
- limited time
- needs to be presentation-friendly
