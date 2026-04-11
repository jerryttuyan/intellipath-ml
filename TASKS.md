# Tasks

## Next
- [x] evaluate METR-LA
- [x] download LargeST CA locally and generate GLA subset
- [x] create LargeST-GLA exploration notebook
- [x] create reusable GLA loader in `src/`
- [x] decide first baseline
- [x] define target variable and starting horizon
- [x] define chronological train/val/test split
- [ ] decide whether to train on all GLA nodes or a smaller subset first
- [ ] expand the baseline run beyond the current 10-node sample
- [ ] add clearer plots / summaries for model comparison
- [ ] decide whether the next model stays tabular or becomes graph-based
- [ ] run repeatable routing comparison (shortest-distance vs current-traffic vs predicted-traffic)
- [ ] add routing metric summaries (avg travel time, p90 travel time, regret) for report

## Open questions
- forecasting only, or routing + forecasting in the presentation?
- is `15 minutes ahead` the main horizon, or should we compare multiple horizons?
- should the first baseline use all 3834 nodes or a narrower slice for speed?
