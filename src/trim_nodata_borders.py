#!/usr/bin/env python3
"""
Credit: John Brandt, https://github.com/wri/tree-verification

Trim 3-band uint8 GeoTIFFs by removing outer borders where all 3 bands are zero.

- Recurses a directory to find .tif / .tiff files
- Computes the minimal bounding window containing any non-zero data
- Writes a cropped copy with a suffix (default "_trim")
- Skips files that are already tightly cropped or contain only zeros

Usage:
  python trim_nodata_borders.py /path/to/folder --suffix "_trim" --inplace
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

def find_nonzero_bbox(ds: rasterio.io.DatasetReader) -> tuple[int,int,int,int] | None:
    """
    Return (row_min, row_max_excl, col_min, col_max_excl) bounding box
    of any pixel where any band != 0. Scans in native block windows to keep memory low.
    Returns None if the entire image is zero.
    """
    if ds.count < 3:
        raise ValueError(f"{ds.name}: expected at least 3 bands, found {ds.count}")
    if ds.dtypes[0] != "uint8":
        raise ValueError(f"{ds.name}: expected uint8 dtype, found {ds.dtypes[0]}")

    row_min = ds.height
    row_max = -1
    col_min = ds.width
    col_max = -1

    # Iterate over block windows (efficient & memory friendly)
    for _, w in ds.block_windows(1):
        # Read the first 3 bands in this window
        data = ds.read(indexes=(1,2,3), window=w, out_dtype="uint8", masked=False)
        # data shape: (3, h, w)
        # True where any band is non-zero
        any_nz = np.any(data != 0, axis=0)
        if not any_nz.any():
            continue

        # Find local bbox inside this window
        rows, cols = np.nonzero(any_nz)
        r0 = w.row_off + rows.min()
        r1 = w.row_off + rows.max() + 1  # exclusive
        c0 = w.col_off + cols.min()
        c1 = w.col_off + cols.max() + 1  # exclusive

        # Update global bbox
        row_min = min(row_min, r0)
        row_max = max(row_max, r1)
        col_min = min(col_min, c0)
        col_max = max(col_max, c1)

    if row_max == -1:
        return None  # all zeros

    return (row_min, row_max, col_min, col_max)


def crop_to_window(
    in_path: Path,
    out_path: Path,
    compress: str = "DEFLATE",
    zlevel: int = 6,
    predictor: int = 2,
    bigtiff: str = "IF_SAFER",
    overview_levels: tuple[int, ...] = (2, 4, 8, 16),
) -> bool:
    """
    Crop input GeoTIFF to the minimal non-zero bbox and write to out_path.
    Returns True if a crop was performed and file written; False if skipped (all-zero or already tight).
    """
    with rasterio.open(in_path) as ds:
        bbox = find_nonzero_bbox(ds)
        if bbox is None:
            print(f"[skip all-zero] {in_path.name}")
            return False

        r0, r1, c0, c1 = bbox
        height = r1 - r0
        width  = c1 - c0

        if height == ds.height and width == ds.width:
            # Already tight; copy or skip
            if in_path.resolve() == out_path.resolve():
                print(f"[already tight] {in_path.name} (no change)")
                return False
            else:
                shutil.copy2(in_path, out_path)
                print(f"[copy (already tight)] {in_path.name} -> {out_path.name}")
                return True

        win = Window.from_slices((r0, r1), (c0, c1))
        transform = rasterio.windows.transform(win, ds.transform)

        profile = ds.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform,
            compress=compress,
            predictor=predictor,  # works with DEFLATE/LZW
            zlevel=zlevel if compress.upper() == "DEFLATE" else None,
            #tiled=True,
            BIGTIFF=bigtiff,
        )

        # Read window and write
        data = ds.read(window=win)  # (bands, H, W)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            # Copy tags
            dst.update_tags(**ds.tags())
            for b in range(1, ds.count + 1):
                dst.update_tags(b, **ds.tags(b))

            # Build overviews for snappier display (optional)
            try:
                factors = [lvl for lvl in overview_levels if lvl > 1 and width//lvl > 0 and height//lvl > 0]
                if factors:
                    dst.build_overviews(factors, Resampling.nearest)
                    dst.update_tags(ns="rio_overview", resampling="nearest")
            except Exception as e:
                print(f"  (overview build skipped: {e})")

    print(f"[cropped] {in_path.name} -> {out_path.name}  ({ds.width}x{ds.height} -> {width}x{height})")
    return True


def main():
    ap = argparse.ArgumentParser(description="Trim borders of 3-band uint8 GeoTIFFs where all bands are zero.")
    ap.add_argument("--folder", type=Path, help="Root folder to search recursively for .tif/.tiff")
    ap.add_argument("--suffix", default="_trim", help="Suffix for output files (before extension). Ignored with --inplace.")
    ap.add_argument("--inplace", action="store_true", help="Replace files in place (safe temp file + move).")
    ap.add_argument("--glob", default="*.tif, *.tiff",
                    help="Comma-separated glob patterns (relative to folder). Default: '**/*.tif,**/*.tiff'")
    args = ap.parse_args()

    patterns = [p.strip() for p in args.glob.split(",") if p.strip()]
    files = []
    for pat in patterns:
        files.extend(sorted(args.folder.glob(pat)))

    if not files:
        print("No GeoTIFFs found.")
        return

    for fp in files:
        if not fp.is_file():
            continue

        # Decide output path
        if args.inplace:
            tmp_out = fp.with_suffix(fp.suffix + ".tmp_trimming")
            wrote = crop_to_window(fp, tmp_out)
            if wrote:
                # Atomic replace
                tmp_out.replace(fp)
            else:
                # Remove temp if created but no crop (already-tight copy case)
                if tmp_out.exists():
                    tmp_out.unlink(missing_ok=True)
        else:
            out_path = fp.with_name(fp.stem + args.suffix + fp.suffix)
            crop_to_window(fp, out_path)


if __name__ == "__main__":
    main()