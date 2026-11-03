#!/usr/bin/env python3
"""Train baseline / HyperCL-Lung ablations for nodule malignancy classification."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tiny3DCNN(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)


class AttrEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


class HyperCLLung(nn.Module):
    def __init__(self, attr_dim: int, emb_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.img_enc = Tiny3DCNN(emb_dim)
        self.attr_enc = AttrEncoder(attr_dim, emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(emb_dim, n_classes)
        self.proj_v = nn.Linear(emb_dim, emb_dim)
        self.proj_t = nn.Linear(emb_dim, emb_dim)
        self.proj_h = nn.Linear(emb_dim, emb_dim)
        self.proj_e = nn.Linear(emb_dim, emb_dim)

    def encode(self, patches, attrs):
        zv = self.img_enc(patches)
        zt = self.attr_enc(attrs)
        h = self.fuse(torch.cat([zv, zt], dim=-1))
        return zv, zt, h

    def forward(self, patches, attrs):
        zv, zt, h = self.encode(patches, attrs)
        return self.cls(h), zv, zt, h


def hyperedge_smooth(feat: torch.Tensor, edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean-aggregate closed neighborhood (self + neighbors)."""
    device = feat.device
    deg = torch.ones(n, device=device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        deg = deg + torch.bincount(dst, minlength=n).float()
        neigh = torch.zeros_like(feat)
        neigh.index_add_(0, dst, feat[src])
        return (feat + neigh) / deg.unsqueeze(-1).clamp_min(1.0)
    return feat


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def load_data(graph_path: Path, device: torch.device):
    graph_path = Path(graph_path).resolve()
    repo_root = graph_path.parent.parent
    data = json.loads(graph_path.read_text())
    patches = []
    for p in data["patch_paths"]:
        pp = Path(p)
        if not pp.is_file():
            pp = repo_root / p
        if not pp.is_file():
            pp = graph_path.parent.parent / "patches" / Path(p).name
        patches.append(np.load(pp))
    x = torch.from_numpy(np.stack(patches)[:, None, ...]).float().to(device)
    attrs = torch.tensor(data["attrs"], dtype=torch.float32, device=device)
    # z-score attrs
    mu, std = attrs.mean(0, keepdim=True), attrs.std(0, keepdim=True).clamp_min(1e-6)
    attrs = (attrs - mu) / std
    y = torch.tensor(data["labels"], dtype=torch.long, device=device)
    splits = np.array(data["splits"])
    edges = torch.tensor(data["edges"], dtype=torch.long, device=device).t().contiguous() if data["edges"] else torch.zeros(2, 0, dtype=torch.long, device=device)
    masks = {
        "train": torch.tensor(splits == "train", device=device),
        "val": torch.tensor(splits == "val", device=device),
        "test": torch.tensor(splits == "test", device=device),
    }
    return x, attrs, y, edges, masks, data


@torch.no_grad()
def eval_split(model, x, attrs, y, mask):
    model.eval()
    logits, _, _, _ = model(x, attrs)
    logits = logits[mask]
    yb = y[mask]
    if yb.numel() == 0:
        return {"acc": 0.0, "bacc": 0.0, "f1": 0.0, "auc": 0.5}
    prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    pred = logits.argmax(-1).cpu().numpy()
    yt = yb.cpu().numpy()
    out = {
        "acc": float(accuracy_score(yt, pred)),
        "bacc": float(balanced_accuracy_score(yt, pred)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(yt, prob)) if len(np.unique(yt)) > 1 else 0.5
    except Exception:
        out["auc"] = 0.5
    return out


def train_one(mode: str, x, attrs, y, edges, masks, args, device):
    model = HyperCLLung(attr_dim=attrs.size(1), emb_dim=args.emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # class weights
    ytr = y[masks["train"]]
    n0 = max(1, int((ytr == 0).sum()))
    n1 = max(1, int((ytr == 1).sum()))
    w = torch.tensor([1.0, n0 / n1], device=device, dtype=torch.float32)
    best_val = -1.0
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits, zv, zt, h = model(x, attrs)
        loss_cls = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=w)

        use_a = mode in {"hga", "full"}
        use_b = mode in {"nhc", "full"}
        loss_a = torch.tensor(0.0, device=device)
        loss_b = torch.tensor(0.0, device=device)
        if use_a or use_b:
            zv_s = hyperedge_smooth(zv, edges, zv.size(0))
            zt_s = hyperedge_smooth(zt, edges, zt.size(0))
            if use_a:
                loss_a = info_nce(model.proj_v(zv_s), model.proj_t(zt_s), args.temperature)
            if use_b:
                u = hyperedge_smooth(h, edges, h.size(0))
                loss_b = info_nce(model.proj_h(h), model.proj_e(u), args.temperature)

        # concat baseline disables contrast
        if mode == "concat":
            loss = loss_cls
        elif mode == "img":
            # image-only: re-encode via zero attrs path already fused; approximate by cls on img branch
            logits_img = model.cls(model.fuse(torch.cat([zv, torch.zeros_like(zt)], dim=-1)))
            loss = F.cross_entropy(logits_img[masks["train"]], y[masks["train"]], weight=w)
        else:
            # Optional multi-task scaling: lambda is the weight relative to CE magnitude,
            # so A and B do not drown the classifier when InfoNCE is large.
            if getattr(args, "scale_to_cls", False):
                def rel(term, lam):
                    if lam == 0:
                        return term * 0
                    scale = loss_cls.detach() / term.detach().clamp_min(1e-6)
                    return lam * term * scale
                loss = loss_cls + rel(loss_a, args.lambda_a) + rel(loss_b, args.lambda_b)
            else:
                loss = loss_cls + args.lambda_a * loss_a + args.lambda_b * loss_b

        opt.zero_grad()
        loss.backward()
        opt.step()

        val = eval_split(model, x, attrs, y, masks["val"])
        history.append({"epoch": epoch, "loss": float(loss.item()), **{f"val_{k}": v for k, v in val.items()}})
        select = getattr(args, "select", "auc_bacc")
        score = val["auc"] if select == "auc" else val["auc"] + val["bacc"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = eval_split(model, x, attrs, y, masks["test"])
    val = eval_split(model, x, attrs, y, masks["val"])
    out = {
        "mode": mode,
        "val": val,
        "test": test,
        "seconds": time.time() - t0,
        "history": history[-5:],
    }
    if getattr(args, "return_model", False):
        out["model"] = model
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=root / "data" / "graph_split.json")
    ap.add_argument("--out", type=Path, default=root / "logs" / "pilot_results.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--lambda-a", type=float, default=0.2)
    ap.add_argument("--lambda-b", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x, attrs, y, edges, masks, meta = load_data(args.graph, device)
    print("N", x.shape, "edges", edges.shape[1], "pos", float(y.float().mean()))
    print("splits", {k: int(v.sum()) for k, v in masks.items()})

    results = []
    for mode in ["img", "concat", "hga", "nhc", "full"]:
        print(f"\n=== train mode={mode} ===")
        r = train_one(mode, x, attrs, y, edges, masks, args, device)
        print("val", r["val"])
        print("test", r["test"])
        results.append(r)

    payload = {
        "meta_counts": meta.get("n_nodules"),
        "series_counts": meta.get("n_series"),
        "results": [{k: v for k, v in r.items() if k != "history"} | {"history_tail": r["history"]} for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("\nWrote", args.out)


if __name__ == "__main__":
    main()
