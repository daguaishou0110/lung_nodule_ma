# HyperCL-Lung

Code and processed data for **HyperCL-Lung**: hypergraph cross-modal alignment and node–hyperedge contrast for lung nodule malignancy classification on LIDC-IDRI / LUNA16.

## Repository layout

```
scripts/                 # training / preprocessing pipeline
  01_parse_lidc.py
  02_extract_patches.py
  03_build_graph.py
  04_train.py            # main train / ablation entry
  05_tune.py
  06_refine.py
  analysis/              # figure / case-analysis helpers
data/
  nodules.csv            # merged nodule table (attributes + labels)
  graph_split.json       # series-wise split, edges, attrs, patch paths
patches/                 # 32³ CT patches (.npy), one file per nodule
logs/                    # saved experiment JSON summaries
```

Raw LUNA16 `.mhd/.raw` volumes are **not** included (large; redistributed by LUNA/TCIA). Pre-extracted `patches/*.npy` are enough to train.

## Data summary

| Item | Value |
|------|--------|
| Source | LUNA16 subsets 0–4 + LIDC-IDRI XML attributes |
| Nodules | 1,134 (580 positive) |
| Series | 384 |
| Split | series-wise 70 / 15 / 15 (train / val / test) |
| Patch size | `32 × 32 × 32`, HU window `[-1000, 400]` → `[-1, 1]` |
| Attributes | texture, spiculation, margin, sphericity, lobulation, subtlety, calcification, internalStructure (malignancy excluded from inputs) |
| Label | mean LIDC malignancy ≥ 3 → positive |

`data/graph_split.json` uses **relative** patch paths (`patches/nXXXXX.npy`).

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- `numpy`, `scikit-learn`
- For rebuilding patches from CT: `SimpleITK`, LIDC XML + LUNA16 volumes

```bash
pip install torch numpy scikit-learn
# optional rebuild:
pip install SimpleITK
```

## Quick start (train on released patches)

```bash
git clone https://github.com/daguaishou0110/lung_nodule_ma.git
cd lung_nodule_ma

# baseline + ablations (img / concat / hga / nhc / full)
python scripts/04_train.py --device cuda:0 --epochs 60

# hyperparameter grid
python scripts/05_tune.py --device cuda:0

# focused multi-seed refine (example)
python scripts/06_refine.py --device cuda:0 --out logs/refine_custom.json \
  --jobs "concat:0:0:0.2:42,full:0.3:1.0:0.1:42,full:0.3:1.0:0.1:7,full:0.3:1.0:0.1:123"
```

Defaults resolve `data/graph_split.json` and write under `logs/` relative to the repo root.

## Rebuild from LIDC / LUNA (optional)

1. Download LIDC-IDRI XML and LUNA16 MetaImage volumes.
2. Point scripts at your local paths:

```bash
python scripts/01_parse_lidc.py --xml-root /path/to/lidc-xml --mhd-root /path/to/luna --out data/nodules.csv
python scripts/02_extract_patches.py --nodules data/nodules.csv --out-dir patches
python scripts/03_build_graph.py --meta patches/patches_meta.csv --out data/graph_split.json
```

`nodules.csv` still contains original `mhd_path` strings from the build machine; they are only needed for step 02.

## Modes in `04_train.py`

| Mode | Meaning |
|------|---------|
| `img` | image encoder only |
| `concat` | image + attribute fusion (host) |
| `hga` | host + hypergraph cross-modal alignment |
| `nhc` | host + node–hyperedge contrast |
| `full` | both attachments |

## Citation / license

If you use this code or processed patches, please cite the accompanying paper (HyperCL-Lung) and the original LIDC-IDRI / LUNA16 resources.  
Processed patches are derived from public datasets; follow TCIA / LUNA usage terms.
