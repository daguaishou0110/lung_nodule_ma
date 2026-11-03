#!/usr/bin/env python3
"""Extract 32x32x32 CT patches around LIDC nodules."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def load_ct(mhd_path: str):
    img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # z,y,x
    return img, arr


def world_to_voxel(img, xyz_world):
    return img.TransformPhysicalPointToIndex(tuple(float(v) for v in xyz_world))


def extract_cube(arr, center_zyx, size=32):
    z, y, x = [int(round(c)) for c in center_zyx]
    half = size // 2
    out = np.zeros((size, size, size), dtype=np.float32)
    for dz in range(size):
        zz = z - half + dz
        if zz < 0 or zz >= arr.shape[0]:
            continue
        for dy in range(size):
            yy = y - half + dy
            if yy < 0 or yy >= arr.shape[1]:
                continue
            x0 = x - half
            x1 = x0 + size
            src0 = max(0, x0)
            src1 = min(arr.shape[2], x1)
            dst0 = src0 - x0
            dst1 = dst0 + (src1 - src0)
            out[dz, dy, dst0:dst1] = arr[zz, yy, src0:src1]
    # HU window to [-1,1]
    out = np.clip(out, -1000, 400)
    out = (out + 1000) / 1400.0 * 2.0 - 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodules", type=Path, default=Path("/tzq2/code/dataset/lung_nodule_mag/data/nodules.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tzq2/code/dataset/lung_nodule_mag/patches"))
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.nodules.open() as f:
        rows = list(csv.DictReader(f))

    cache: dict[str, tuple] = {}
    meta_path = args.out_dir / "patches_meta.csv"
    kept = []
    for row in tqdm(rows, desc="extract"):
        mhd = row["mhd_path"]
        if mhd not in cache:
            cache[mhd] = load_ct(mhd)
        img, arr = cache[mhd]
        # LIDC edge maps are in voxel xy; z is world imageZposition
        # Use (cx,cy) as voxel, cz as world z.
        spacing = img.GetSpacing()  # x,y,z
        origin = img.GetOrigin()
        # convert cz world -> voxel z
        # Physical point (origin-based): for z only
        try:
            # cx,cy appear to be voxel indices in XML edgeMap
            vx = int(round(float(row["cx"])))
            vy = int(round(float(row["cy"])))
            # cz is world z from imageZposition
            wz = float(row["cz"])
            vz = int(round((wz - origin[2]) / spacing[2]))
            if abs(vz) > arr.shape[0] * 2:
                # fallback: treat cz as voxel
                vz = int(round(float(row["cz"])))
        except Exception:
            continue
        if not (0 <= vz < arr.shape[0] and 0 <= vy < arr.shape[1] and 0 <= vx < arr.shape[2]):
            # try treating all as world
            try:
                ix, iy, iz = world_to_voxel(img, (float(row["cx"]), float(row["cy"]), float(row["cz"])))
                vx, vy, vz = ix, iy, iz
            except Exception:
                continue
        if not (0 <= vz < arr.shape[0] and 0 <= vy < arr.shape[1] and 0 <= vx < arr.shape[2]):
            continue
        cube = extract_cube(arr, (vz, vy, vx), size=args.size)
        out_path = args.out_dir / f"{row['nodule_id']}.npy"
        np.save(out_path, cube)
        row = dict(row)
        row["patch_path"] = str(out_path)
        row["vx"], row["vy"], row["vz"] = vx, vy, vz
        kept.append(row)

    if not kept:
        raise SystemExit("No patches extracted.")
    fields = list(kept[0].keys())
    with meta_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(f"Saved {len(kept)} patches -> {args.out_dir}")
    print(f"Meta -> {meta_path}")


if __name__ == "__main__":
    main()
