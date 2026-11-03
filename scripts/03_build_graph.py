#!/usr/bin/env python3
"""Build series-level nodule graph splits for HyperCL-Lung."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# NOTE: do NOT include malignancy — it defines the label (mean >= 3).
ATTR_COLS = [
    "texture",
    "spiculation",
    "margin",
    "sphericity",
    "lobulation",
    "subtlety",
    "calcification",
    "internalStructure",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--meta",
        type=Path,
        default=Path("/tzq2/code/dataset/lung_nodule_mag/patches/patches_meta.csv"),
    )
    ap.add_argument("--out", type=Path, default=Path("/tzq2/code/dataset/lung_nodule_mag/data/graph_split.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    for c in ATTR_COLS:
        if c not in df.columns:
            df[c] = 3.0
        df[c] = df[c].fillna(df[c].median())

    series = df["series_uid"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(series)
    n = len(series)
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(1, int(n * args.val_ratio))
    train_s = set(series[:n_train])
    val_s = set(series[n_train : n_train + n_val])
    test_s = set(series[n_train + n_val :])
    if not test_s:
        # move one from train
        moved = series[0]
        train_s.remove(moved)
        test_s.add(moved)

    def split_of(uid: str) -> str:
        if uid in train_s:
            return "train"
        if uid in val_s:
            return "val"
        return "test"

    df["split"] = df["series_uid"].map(split_of)

    # adjacency: fully connect nodules within the same series
    id_list = df["nodule_id"].tolist()
    id2i = {nid: i for i, nid in enumerate(id_list)}
    edges = []
    for _, g in df.groupby("series_uid"):
        ids = g["nodule_id"].tolist()
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                edges.append([id2i[a], id2i[b]])

    payload = {
        "nodule_ids": id_list,
        "labels": df["label"].astype(int).tolist(),
        "splits": df["split"].tolist(),
        "series_uid": df["series_uid"].tolist(),
        "patch_paths": df["patch_path"].tolist(),
        "attrs": df[ATTR_COLS].astype(float).values.tolist(),
        "attr_cols": ATTR_COLS,
        "edges": edges,
        "n_series": {
            "train": len(train_s),
            "val": len(val_s),
            "test": len(test_s),
        },
        "n_nodules": {
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print("series", payload["n_series"])
    print("nodules", payload["n_nodules"])
    print("edges", len(edges))
    print("pos rate", float(np.mean(payload["labels"])))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
