#!/usr/bin/env python3
"""Parse LIDC XML annotations into a per-nodule CSV."""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NS = {"lidc": "http://www.nih.gov"}


def local(tag: str) -> str:
    return tag.split("}")[-1]


def parse_xml(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    series_uid = None
    for el in root.iter():
        if local(el.tag) == "SeriesInstanceUid" and el.text:
            series_uid = el.text.strip()
            break
    if not series_uid:
        return []

    nodules: list[dict] = []
    for el in root.iter():
        if local(el.tag) != "unblindedReadNodule":
            continue
        chars: dict[str, float] = {}
        xs, ys, zs = [], [], []
        for child in el.iter():
            t = local(child.tag)
            if t in {
                "malignancy",
                "texture",
                "spiculation",
                "margin",
                "sphericity",
                "lobulation",
                "subtlety",
                "calcification",
                "internalStructure",
            } and child.text and child.text.strip().isdigit():
                chars[t] = float(child.text.strip())
            if t == "xCoord" and child.text:
                xs.append(float(child.text))
            if t == "yCoord" and child.text:
                ys.append(float(child.text))
            if t == "imageZposition" and child.text:
                zs.append(float(child.text))
        if not xs or "malignancy" not in chars:
            continue
        nodules.append(
            {
                "series_uid": series_uid,
                "xml_path": str(path),
                "cx": sum(xs) / len(xs),
                "cy": sum(ys) / len(ys),
                "cz": sum(zs) / len(zs) if zs else 0.0,
                "n_edge": len(xs),
                **chars,
            }
        )
    return nodules


def cluster_reads(reads: list[dict], dist_xy: float = 6.0, dist_z: float = 5.0) -> list[dict]:
    """Greedy merge of nearby radiologist reads into one nodule."""
    used = [False] * len(reads)
    out: list[dict] = []
    for i, a in enumerate(reads):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j in range(i + 1, len(reads)):
            if used[j]:
                continue
            b = reads[j]
            if abs(a["cx"] - b["cx"]) <= dist_xy and abs(a["cy"] - b["cy"]) <= dist_xy and abs(a["cz"] - b["cz"]) <= dist_z:
                group.append(b)
                used[j] = True
        keys = [
            "malignancy",
            "texture",
            "spiculation",
            "margin",
            "sphericity",
            "lobulation",
            "subtlety",
            "calcification",
            "internalStructure",
        ]
        row = {
            "series_uid": a["series_uid"],
            "cx": sum(g["cx"] for g in group) / len(group),
            "cy": sum(g["cy"] for g in group) / len(group),
            "cz": sum(g["cz"] for g in group) / len(group),
            "n_reads": len(group),
            "n_edge": int(sum(g["n_edge"] for g in group) / len(group)),
        }
        for k in keys:
            vals = [g[k] for g in group if k in g]
            row[k] = sum(vals) / len(vals) if vals else float("nan")
        # binary label: mean malignancy >= 3
        row["label"] = int(row["malignancy"] >= 3.0) if row["malignancy"] == row["malignancy"] else 0
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml-root",
        type=Path,
        default=Path("/tzq2/code/dataset/luna16_subset0/extracted/SCFNet_Data/tcia-lidc-xml"),
    )
    ap.add_argument(
        "--subset-dirs",
        nargs="+",
        default=[
            "/tzq2/code/dataset/luna16_subset0/extracted/SCFNet_Data/subset0",
            "/tzq2/code/dataset/luna16_subset1_4/extracted/subset1",
            "/tzq2/code/dataset/luna16_subset1_4/extracted/subset2",
            "/tzq2/code/dataset/luna16_subset1_4/extracted/subset3",
            "/tzq2/code/dataset/luna16_subset1_4/extracted/subset4",
        ],
    )
    ap.add_argument("--out", type=Path, default=Path("/tzq2/code/dataset/lung_nodule_mag/data/nodules.csv"))
    ap.add_argument("--pilot-subset0-only", action="store_true")
    args = ap.parse_args()

    available_uids = set()
    uid_to_mhd: dict[str, str] = {}
    subset_dirs = args.subset_dirs
    if args.pilot_subset0_only:
        subset_dirs = [subset_dirs[0]]
    for d in subset_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for mhd in p.glob("*.mhd"):
            uid = mhd.stem
            available_uids.add(uid)
            uid_to_mhd[uid] = str(mhd)

    by_series: dict[str, list[dict]] = defaultdict(list)
    xml_files = sorted(args.xml_root.rglob("*.xml"))
    for xp in xml_files:
        for row in parse_xml(xp):
            if row["series_uid"] in available_uids:
                by_series[row["series_uid"]].append(row)

    rows: list[dict] = []
    nid = 0
    for uid, reads in sorted(by_series.items()):
        for nod in cluster_reads(reads):
            nod["nodule_id"] = f"n{nid:05d}"
            nod["mhd_path"] = uid_to_mhd[uid]
            rows.append(nod)
            nid += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("No nodules matched available CT series.")
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(r["label"] for r in rows)
    print(f"Wrote {len(rows)} nodules from {len(by_series)} series -> {args.out}")
    print(f"label=1: {n_pos}, label=0: {len(rows) - n_pos}")


if __name__ == "__main__":
    main()
