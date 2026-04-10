# LargeST GLA Setup Note

This is a minimal note for evaluating a switch from METR-LA to the official LargeST Greater Los Angeles (GLA) subset.

## Official source

- Official LargeST repository: https://github.com/liuxu77/LargeST
- Official CA dataset host referenced by the repo: https://www.kaggle.com/datasets/liuxu77/largest

## What the official repo says

The official workflow is:

1. Download the California (CA) dataset from Kaggle.
2. Process the CA history files using the notebook in the official repo's `data/ca/`.
3. Derive the GLA subset using the official repo's `data/gla/generate_gla_dataset.ipynb`.
4. Generate training artifacts with:

```bash
python data/generate_data_for_training.py --dataset gla --years 2019
```

The official script writes processed files like:

- `gla/2019/his.npz`
- `gla/2019/idx_train.npy`
- `gla/2019/idx_val.npy`
- `gla/2019/idx_test.npy`

The same script expects a raw GLA HDF5 file named like:

- `gla/gla_his_2019.h5`

## Suggested local layout for this repo

To keep this repo organized without changing project code broadly, a reasonable local layout is:

```text
data/raw/LargeST/ca/
data/raw/LargeST/gla/
data/raw/LargeST/gla/2019/
```

Likely files to expect after following the official process:

- `data/raw/LargeST/gla/gla_his_2019.h5`
- `data/raw/LargeST/gla/gla_meta.csv`
- `data/raw/LargeST/gla/gla_rn_adj.npy`
- `data/raw/LargeST/gla/2019/his.npz`
- `data/raw/LargeST/gla/2019/idx_train.npy`
- `data/raw/LargeST/gla/2019/idx_val.npy`
- `data/raw/LargeST/gla/2019/idx_test.npy`

## Important note

The official repo documents how to derive GLA from CA, but the exact generated filenames for metadata and adjacency should be confirmed after running the official notebook once. The feasibility notebook in this repo is intentionally flexible and will scan for likely GLA raw, metadata, and adjacency files rather than assuming a single exact filename.

For collaboration, do not commit the raw or generated LargeST dataset files to git. They are very large and are better handled as local-only data with a documented download / generation workflow. Commit the analysis notebooks and setup instructions instead, and have teammates fetch the CA data from Kaggle and generate the GLA subset locally.
