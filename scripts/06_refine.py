#!/usr/bin/env python3
"""Focused retrain: longer epochs, a few seeds, configs near the first-round winners."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_train_mod():
    path = Path(__file__).with_name("04_train.py")
    spec = importlib.util.spec_from_file_location("train04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=Path("/tzq2/code/dataset/lung_nodule_mag/data/graph_split.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--jobs", required=True, help="mode:la:lb:temp:seed comma-separated")
    args = ap.parse_args()

    mod = load_train_mod()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    x, attrs, y, edges, masks, _ = mod.load_data(args.graph, device)

    class Cfg:
        pass

    rows = []
    if args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])

    for job in args.jobs.split(","):
        parts = job.split(":")
        mode, la, lb, temp, seed = parts[:5]
        la, lb, temp, seed = float(la), float(lb), float(temp), int(seed)
        cfg = Cfg()
        cfg.emb_dim = 64
        cfg.lr = 1e-3
        cfg.epochs = args.epochs
        cfg.temperature = temp
        cfg.lambda_a = la
        cfg.lambda_b = lb
        cfg.select = "auc"
        cfg.scale_to_cls = len(parts) > 5 and parts[5] == "rel"
        mod.set_seed(seed)
        r = mod.train_one(mode, x, attrs, y, edges, masks, cfg, device)
        row = {
            "mode": mode, "lambda_a": la, "lambda_b": lb, "temperature": temp,
            "epochs": args.epochs, "seed": seed, "scale_to_cls": cfg.scale_to_cls,
            "val_auc": r["val"]["auc"], "val_bacc": r["val"]["bacc"],
            "test_auc": r["test"]["auc"], "test_bacc": r["test"]["bacc"],
            "test_acc": r["test"]["acc"], "test_f1": r["test"]["f1"],
        }
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2))
        print(
            f"{mode:6} la={la:.3f} lb={lb:.3f} t={temp:.2f} seed={seed} "
            f"val={row['val_auc']:.4f} test={row['test_auc']:.4f}",
            flush=True,
        )
    print("wrote", args.out, "n", len(rows))


if __name__ == "__main__":
    main()
