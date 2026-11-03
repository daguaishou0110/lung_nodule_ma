#!/usr/bin/env python3
"""Figures for the Frontiers HyperCL-Lung manuscript."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_CANDIDATES = [
    Path("/tzq2/code/dataset/lung_nodule_mag"),
    Path(__file__).resolve().parents[2],
]
ROOT = next((p for p in _CANDIDATES if (p / "logs").exists()), _CANDIDATES[-1])
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})
C = {
    "concat": "#4C78A8",
    "hga": "#F58518",
    "nhc": "#54A24B",
    "full": "#E45756",
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig_method():
    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#222"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    facecolor=fc, edgecolor=ec, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#333", linewidth=1.1))

    box(2, 28, 18, 12, "CT patch\n32×32×32", "#D6EAF8")
    box(2, 6, 18, 12, "LIDC attributes\n(no malignancy)", "#FDEBD0")
    box(26, 28, 16, 12, "3D CNN\n$z^v$", "#AED6F1")
    box(26, 6, 16, 12, "MLP\n$z^t$", "#FAD7A0")
    box(48, 17, 18, 12, "Same-series\nhyperedge\nsmooth", "#E8DAEF")
    box(72, 28, 18, 10, "Module A\nInfoNCE\n$z^v_s \\leftrightarrow z^t_s$", "#F5CBA7")
    box(72, 6, 18, 10, "Module B\nInfoNCE\n$H \\leftrightarrow u_e$", "#ABEBC6")
    box(94, 17, 16, 12, "Classifier\nmalignancy\n≥ 3", "#F5B7B1")
    arrow(20, 34, 26, 34)
    arrow(20, 12, 26, 12)
    arrow(42, 34, 48, 26)
    arrow(42, 12, 48, 20)
    arrow(66, 26, 72, 33)
    arrow(66, 20, 72, 11)
    arrow(90, 33, 94, 26)
    arrow(90, 11, 94, 20)
    ax.text(56, 42, "HyperCL-Lung", ha="center", fontsize=13, fontweight="bold")
    save(fig, "fig_method")


def fig_ablation():
    # 3 seeds, order 42, 7, 123
    val = {
        "Concat": [0.8080, 0.7930, 0.8014],
        "HGA": [0.8126, 0.8116, 0.8097],
        "NHC": [0.8251, 0.8255, 0.8216],
        "Full": [0.8297, 0.8278, 0.8229],
    }
    test = {
        "Concat": [0.8624, 0.8833, 0.8895],
        "HGA": [0.8899, 0.8932, 0.8873],
        "NHC": [0.8868, 0.8944, 0.8826],
        "Full": [0.9005, 0.8988, 0.8990],
    }
    colors = [C["concat"], C["hga"], C["nhc"], C["full"]]
    labels = list(val)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, data, title in zip(axes, [val, test], ["Validation AUC", "Test AUC"]):
        means = [np.mean(data[k]) for k in labels]
        stds = [np.std(data[k], ddof=1) for k in labels]
        x = np.arange(len(labels))
        ax.axhline(means[0], color="#888", ls="--", lw=0.8, zorder=0)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#666", elinewidth=1.1, capsize=3, zorder=2)
        ax.scatter(x, means, s=70, c=colors, zorder=3, edgecolors="white", linewidths=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.76, 0.93)
        ax.set_title(title)
        for i, m in enumerate(means):
            ax.text(i, m + stds[i] + 0.004, f"{m:.3f}", ha="center", fontsize=8)
    axes[0].set_ylabel("AUC")
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_sensitivity():
    singles = json.loads((ROOT / "logs/tune_singles.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))

    def curve(mode, lam_key, ax, xlabel):
        rows = [r for r in singles if r["mode"] == mode]
        temps = sorted(set(r["temperature"] for r in rows))
        for t, color in zip(temps, ["#4C78A8", "#F58518", "#54A24B"]):
            sub = sorted([r for r in rows if r["temperature"] == t], key=lambda r: r[lam_key])
            ax.plot([r[lam_key] for r in sub], [r["val_auc"] for r in sub],
                    marker="o", label=f"τ={t:g}", color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Validation AUC")
        ax.legend(frameon=False, fontsize=8)

    curve("hga", "lambda_a", axes[0], r"$\lambda_A$ (HGA only)")
    curve("nhc", "lambda_b", axes[1], r"$\lambda_B$ (NHC only)")
    fig.tight_layout()
    save(fig, "fig_sensitivity")


def fig_heatmap():
    full = json.loads((ROOT / "logs/tune_full.json").read_text(encoding="utf-8"))["rows"]
    temps = [0.1, 0.2]
    las = sorted(set(r["lambda_a"] for r in full))
    lbs = sorted(set(r["lambda_b"] for r in full))
    grids = []
    for t in temps:
        grid = np.full((len(lbs), len(las)), np.nan)
        for r in full:
            if abs(r["temperature"] - t) > 1e-9:
                continue
            i = lbs.index(r["lambda_b"])
            j = las.index(r["lambda_a"])
            grid[i, j] = r["val_auc"]
        grids.append(grid)
    finite = np.concatenate([g[np.isfinite(g)] for g in grids])
    vmin = float(np.floor(finite.min() * 200) / 200)
    vmax = float(np.ceil(finite.max() * 200) / 200)
    mid = 0.5 * (vmin + vmax)

    # Give the right panel a bit more width for the colorbar; hide its y tick labels
    # so they cannot stack on the left-column cell numbers.
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.18},
    )
    for ai, (ax, t, grid) in enumerate(zip(axes, temps, grids)):
        im = ax.imshow(
            grid, origin="lower", cmap="YlOrRd", vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        n = len(las)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{v:g}" for v in las])
        ax.set_yticks(range(n))
        if ai == 0:
            ax.set_yticklabels([f"{v:g}" for v in lbs])
            ax.set_ylabel(r"$\lambda_{\mathrm{cl}}$")
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.set_ylabel("")
        ax.set_xlabel(r"$\lambda_{\mathrm{hg}}$")
        ax.set_title(rf"Full model, $\tau={t:g}$", pad=6)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0)
        for i in range(n):
            for j in range(n):
                v = grid[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color=("white" if v >= mid else "#111111"),
                    clip_on=False, zorder=3,
                )
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.10, label="Val AUC")
    save(fig, "fig_heatmap")


def fig_patches():
    meta = pd.read_csv(ROOT / "patches/patches_meta.csv")
    pos = meta[meta["label"] == 1].head(4)
    neg = meta[meta["label"] == 0].head(4)
    fig, axes = plt.subplots(2, 4, figsize=(8.2, 4.2))
    for row, (block, name) in enumerate([(neg, "Benign-leaning"), (pos, "Malignant-leaning")]):
        for col, (_, rec) in enumerate(block.iterrows()):
            vol = np.load(rec["patch_path"])
            sl = vol[vol.shape[0] // 2]
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_title(f"mal. {float(rec['malignancy']):.1f}", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_patches")


if __name__ == "__main__":
    fig_method()
    fig_ablation()
    fig_sensitivity()
    fig_heatmap()
    fig_patches()
    print("wrote", FIG)
