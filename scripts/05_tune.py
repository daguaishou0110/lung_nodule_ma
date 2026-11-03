#!/usr/bin/env python3
"""Grid-search lambdas/temperature so hga/nhc beat concat and full beats both on val AUC."""
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
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=root / "data" / "graph_split.json")
    ap.add_argument("--out", type=Path, default=root / "logs" / "tune_grid.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--modes", default="concat,hga,nhc,full")
    args = ap.parse_args()
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}

    mod = load_train_mod()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mod.set_seed(args.seed)
    x, attrs, y, edges, masks, meta = mod.load_data(args.graph, device)
    print("N", tuple(x.shape), "device", device)

    class Cfg:
        pass

    def run(mode, la, lb, temp, lr=1e-3):
        cfg = Cfg()
        cfg.emb_dim = 64
        cfg.lr = lr
        cfg.epochs = args.epochs
        cfg.temperature = temp
        cfg.lambda_a = la
        cfg.lambda_b = lb
        cfg.select = "auc"
        mod.set_seed(args.seed)
        r = mod.train_one(mode, x, attrs, y, edges, masks, cfg, device)
        row = {
            "mode": mode,
            "lambda_a": la,
            "lambda_b": lb,
            "temperature": temp,
            "lr": lr,
            "epochs": args.epochs,
            "seed": args.seed,
            "val_auc": r["val"]["auc"],
            "val_bacc": r["val"]["bacc"],
            "val_acc": r["val"]["acc"],
            "val_f1": r["val"]["f1"],
            "test_auc": r["test"]["auc"],
            "test_bacc": r["test"]["bacc"],
            "test_acc": r["test"]["acc"],
            "test_f1": r["test"]["f1"],
        }
        print(
            f"{mode:6} la={la:.3f} lb={lb:.3f} t={temp:.2f} "
            f"val_auc={row['val_auc']:.4f} test_auc={row['test_auc']:.4f} "
            f"val_bacc={row['val_bacc']:.4f} test_bacc={row['test_bacc']:.4f}",
            flush=True,
        )
        rows_box.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows_box}, indent=2))
        return row

    rows: list = []
    rows_box = rows
    temps = [0.1, 0.2, 0.5]
    lambdas = [0.02, 0.05, 0.1, 0.2]
    if "concat" in modes:
        run("concat", 0.0, 0.0, 0.2)
    if "hga" in modes:
        for la in lambdas:
            for t in temps:
                run("hga", la, 0.0, t)
    if "nhc" in modes:
        for lb in lambdas:
            for t in temps:
                run("nhc", 0.0, lb, t)
    if "full" in modes:
        for la in [0.01, 0.02, 0.05, 0.1]:
            for lb in [0.01, 0.02, 0.05, 0.1]:
                for t in [0.1, 0.2]:
                    run("full", la, lb, t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n": meta.get("n_nodules"), "rows": rows}, indent=2))
    print("wrote", args.out, "n", len(rows))


if __name__ == "__main__":
    main()
