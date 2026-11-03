#!/usr/bin/env python3
"""Paired host-vs-full dump: outcome mix, confusion, t-SNE, error causes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

ROOT = Path("/tzq2/code/dataset/lung_nodule_mag")
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_train():
    path = ROOT / "scripts" / "04_train.py"
    spec = importlib.util.spec_from_file_location("train04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Cfg:
    emb_dim = 64
    lr = 1e-3
    epochs = 60
    select = "auc"
    scale_to_cls = True


def run(mod, mode, la, lb, temp, x, attrs, y, edges, masks, device):
    cfg = Cfg()
    cfg.temperature = temp
    cfg.lambda_a = la
    cfg.lambda_b = lb
    cfg.return_model = True
    mod.set_seed(42)
    r = mod.train_one(mode, x, attrs, y, edges, masks, cfg, device)
    model = r.pop("model")
    model.eval()
    with torch.no_grad():
        logits, zv, zt, h = model(x, attrs)
    return {
        "pred": logits.argmax(-1).cpu().numpy(),
        "prob": F.softmax(logits, dim=-1)[:, 1].cpu().numpy(),
        "zv": zv.cpu().numpy(),
        "zt": zt.cpu().numpy(),
        "h": h.cpu().numpy(),
        "test": r["test"],
        "val": r["val"],
    }


def neighbor_purity(labels, edges, n):
    pur = np.ones(n, dtype=np.float32)
    if edges.numel() == 0:
        return pur
    src = edges[0].cpu().numpy()
    dst = edges[1].cpu().numpy()
    buckets = [[] for _ in range(n)]
    for s, d in zip(src, dst):
        if s != d:
            buckets[int(d)].append(int(s))
    for i, neigh in enumerate(buckets):
        if not neigh:
            pur[i] = 1.0
        else:
            pur[i] = np.mean(labels[neigh] == labels[i])
    return pur


def main():
    mod = load_train()
    device = torch.device("cuda:0")
    x, attrs, y, edges, masks, _ = mod.load_data(ROOT / "data" / "graph_split.json", device)
    y_np = y.cpu().numpy()
    test = masks["test"].cpu().numpy()
    host = run(mod, "concat", 0.0, 0.0, 0.2, x, attrs, y, edges, masks, device)
    full = run(mod, "full", 0.3, 1.0, 0.1, x, attrs, y, edges, masks, device)
    print("host", host["test"])
    print("full", full["test"])

    yt = y_np[test]
    ph = host["pred"][test]
    pf = full["pred"][test]
    both_ok = (ph == yt) & (pf == yt)
    fixed = (ph != yt) & (pf == yt)
    broken = (ph == yt) & (pf != yt)
    both_bad = (ph != yt) & (pf != yt)
    counts = {
        "both_correct": int(both_ok.sum()),
        "fixed": int(fixed.sum()),
        "broken": int(broken.sum()),
        "both_wrong": int(both_bad.sum()),
        "n_test": int(test.sum()),
    }
    print("outcome", counts)

    pur = neighbor_purity(y_np, edges, len(y_np))
    zv = full["zv"]
    zt = full["zt"]
    cos = np.sum(zv * zt, axis=1) / (np.linalg.norm(zv, axis=1) * np.linalg.norm(zt, axis=1) + 1e-8)
    test_idx = np.where(test)[0]
    correct_full = test_idx[pf == yt]
    med = float(np.median(cos[correct_full])) if len(correct_full) else 0.0
    miss = test_idx[pf != yt]
    cause = {"heterophily": 0, "mismatch": 0, "both": 0, "confusion": 0}
    for i in miss:
        het = pur[i] < 0.5
        mis = cos[i] < med
        if het and mis:
            cause["both"] += 1
        elif het:
            cause["heterophily"] += 1
        elif mis:
            cause["mismatch"] += 1
        else:
            cause["confusion"] += 1
    print("cause", cause, "median_cos", med)

    # correct vs wrong factors
    ok = test_idx[pf == yt]
    factors = {
        "purity_correct": float(pur[ok].mean()) if len(ok) else 0,
        "purity_wrong": float(pur[miss].mean()) if len(miss) else 0,
        "cos_correct": float(cos[ok].mean()) if len(ok) else 0,
        "cos_wrong": float(cos[miss].mean()) if len(miss) else 0,
    }
    print("factors", factors)

    summary = {
        "host_test": host["test"],
        "full_test": full["test"],
        "outcome": counts,
        "cause": cause,
        "factors": factors,
        "note": "seed 42 dump of concat vs full (lambda_a=0.3, lambda_b=1, tau=0.1, scale_to_cls)",
    }
    (FIG / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "savefig.bbox": "tight"})

    # outcome flow: host decision -> joint decision
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    both_c, fixed, broken, both_w = counts["both_correct"], counts["fixed"], counts["broken"], counts["both_wrong"]
    left_order = [("both_wrong", both_w, "#BAB0AC"), ("fixed", fixed, "#54A24B"),
                  ("broken", broken, "#E45756"), ("both_correct", both_c, "#4C78A8")]
    right_order = [("both_wrong", both_w, "#BAB0AC"), ("broken", broken, "#E45756"),
                   ("fixed", fixed, "#54A24B"), ("both_correct", both_c, "#4C78A8")]
    def stack(order):
        y, pos = 0.0, {}
        for key, val, _ in order:
            pos[key] = (y, y + val)
            y += val
        return pos, y
    left_pos, total = stack(left_order)
    right_pos, _ = stack(right_order)
    t = np.linspace(0, 1, 40)
    for key, val, color in left_order:
        y0a, y0b = left_pos[key]
        y1a, y1b = right_pos[key]
        upper = (1 - t) ** 3 * y0b + 3 * (1 - t) ** 2 * t * y0b + 3 * (1 - t) * t ** 2 * y1b + t ** 3 * y1b
        lower = (1 - t) ** 3 * y0a + 3 * (1 - t) ** 2 * t * y0a + 3 * (1 - t) * t ** 2 * y1a + t ** 3 * y1a
        ax.fill(np.concatenate([t, t[::-1]]), np.concatenate([upper, lower[::-1]]),
                color=color, alpha=0.8, lw=0)
        mid = (1 - 0.16) ** 3 * ((y0a + y0b) / 2) + 3 * (1 - 0.16) ** 2 * 0.16 * ((y0a + y0b) / 2) + 3 * (1 - 0.16) * 0.16 ** 2 * ((y1a + y1b) / 2) + 0.16 ** 3 * ((y1a + y1b) / 2)
        ax.text(0.16, mid, str(val), ha="center", va="center", fontsize=8, color="#222")
    ax.plot([0, 0], [0, total], color="#333", lw=6, solid_capstyle="butt")
    ax.plot([1, 1], [0, total], color="#333", lw=6, solid_capstyle="butt")
    ax.text(-0.04, total + 4, "Host", ha="center", fontsize=11)
    ax.text(1.04, total + 4, "Joint", ha="center", fontsize=11)
    ax.text(-0.06, (left_pos["both_correct"][0] + left_pos["broken"][1]) / 2, "correct", ha="right", va="center", fontsize=8)
    ax.text(-0.06, (left_pos["both_wrong"][0] + left_pos["fixed"][1]) / 2, "wrong", ha="right", va="center", fontsize=8)
    ax.text(1.06, (right_pos["both_correct"][0] + right_pos["fixed"][1]) / 2, "correct", ha="left", va="center", fontsize=8)
    ax.text(1.06, (right_pos["both_wrong"][0] + right_pos["broken"][1]) / 2, "wrong", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-6, total + 14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_outcome.pdf")
    plt.close()

    # cause donut
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ck = ["Heterophily", "Mismatch", "Both", "Confusion"]
    cv = [cause["heterophily"], cause["mismatch"], cause["both"], cause["confusion"]]
    colors = ["#F58518", "#4C78A8", "#E45756", "#BAB0AC"]
    wedges, _ = ax.pie(cv, startangle=90, colors=colors, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, [f"{n}  {v}" for n, v in zip(ck, cv)], frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.text(0, 0, f"{sum(cv)}\nerrors", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cause.pdf")
    plt.close()

    # confusion: dedicated colorbar column so annotations on the right panel stay clear
    fig = plt.figure(figsize=(7.6, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.35,
                          left=0.08, right=0.92, bottom=0.18, top=0.88)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    im = None
    for ax, pred, title in zip(axes, [ph, pf], ["Concat", "Full"]):
        cm = confusion_matrix(yt, pred, labels=[0, 1]).astype(float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues", aspect="equal", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                        color=("white" if v >= 0.55 else "black"), clip_on=False)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
    fig.colorbar(im, cax=cax)
    fig.savefig(FIG / "fig_confusion.pdf")
    fig.savefig(FIG / "fig_confusion.png")
    plt.close()

    # t-SNE on test fused states
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, emb, title in zip(axes, [host["h"][test], full["h"][test]], ["Concat", "Full"]):
        xy = TSNE(n_components=2, perplexity=20, init="pca", random_state=42).fit_transform(emb)
        for lab, color, name in [(0, "#4C78A8", "Neg"), (1, "#E45756", "Pos")]:
            m = yt == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=12, c=color, label=name, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tsne.pdf")
    plt.close()

    # factors as two slopes; purity and cosine are not on one scale
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
    panels = [
        ("Neighbor purity", factors["purity_correct"], factors["purity_wrong"], (0.45, 0.9)),
        ("Image--attr cosine", factors["cos_correct"], factors["cos_wrong"], (-0.01, 0.06)),
    ]
    for ax, (title, ok_v, bad_v, ylim) in zip(axes, panels):
        ax.plot([0, 1], [ok_v, bad_v], color="#333", lw=1.4)
        ax.scatter([0, 1], [ok_v, bad_v], s=64, c=["#54A24B", "#E45756"], zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Wrong"])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.text(0, ok_v, f" {ok_v:.2f}", va="bottom", ha="left", fontsize=8)
        ax.text(1, bad_v, f"{bad_v:.2f} ", va="bottom", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_factors.pdf")
    plt.close()
    print("wrote figures")


if __name__ == "__main__":
    main()
