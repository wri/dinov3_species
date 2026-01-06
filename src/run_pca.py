#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
import torch.nn.functional as F

from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16


WINDOW_PX   = 768          # tile size (px)
STRIDE_PX   = 256          # overlap stride (px)
PATCH_SIZE  = 16
N_PC        = 3
TARGET_SAMPLE = 20_000

READ_MASKED = False
EMPTY_FRAC_THRESH = 0.5

# --- vitb16 ---
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
REPO_DIR = "../dinov3"
BACKBONE = "vitb16"
URL = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# --- vitl16 ---
# MEAN = (0.430, 0.411, 0.296)
# STD  = (0.213, 0.156, 0.143)
# REPO_DIR = "../dinov3"
# BACKBONE = "vitl16"
# URL = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

DEVICE = "mps"

# ===================== HELPERS =====================
def _axis_starts(full_len: int, win: int, stride: int) -> List[int]:
    """
    Start positions ensuring end coverage.
    If full_len <= win: [0]
    Else: regular stride starts + a final start at (full_len - win) if needed.
    """
    if full_len <= win:
        return [0]
    starts = list(range(0, full_len - win + 1, stride))
    last = full_len - win
    if starts[-1] != last:
        starts.append(last)
    return starts

def build_overlapping_windows_with_edges(src, win_px: int, stride_px: int) -> List[Window]:
    h, w = src.height, src.width
    ys = _axis_starts(h, win_px, stride_px)
    xs = _axis_starts(w, win_px, stride_px)
    wins = []
    for y in ys:
        for x in xs:
            win_h = min(win_px, h - y)
            win_w = min(win_px, w - x)
            if win_h > 0 and win_w > 0:
                wins.append(Window(x, y, win_w, win_h))
    return wins

def resize_transform(pil: Image.Image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Top-left crop to multiples of patch_size with no resampling
    """
    w, h = pil.size
    out_h = (h // patch_size) * patch_size
    out_w = (w // patch_size) * patch_size
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Window too small after patch crop: in=({w},{h}) out=({out_w},{out_h})")
    pil = pil.crop((0, 0, out_w, out_h))
    return TF.to_tensor(pil)  # CHW float32 [0,1]

def window_is_empty(src, win: Window, empty_frac_thresh: float = EMPTY_FRAC_THRESH) -> bool:
    arr = src.read([1, 2, 3], window=win, masked=READ_MASKED)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0)
    # empty if all three bands are 0 at a pixel
    empty_mask = np.all(arr == 0, axis=0)
    empty_frac = float(empty_mask.mean())
    return empty_frac > empty_frac_thresh

def window_to_tensor(src, win: Window) -> torch.Tensor:
    """
    Read window -> PIL -> crop to patch multiple -> normalize
    Returns (3, H*, W*) where H*,W* are multiples of PATCH_SIZE
    """
    arr = src.read([1, 2, 3], window=win, masked=READ_MASKED)  # (3,H,W)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0)
    hwc = np.transpose(arr, (1, 2, 0))  # (H,W,3)

    # map dtype to uint8 for PIL
    if hwc.dtype == np.uint8:
        u8 = hwc
    elif hwc.dtype == np.uint16:
        u8 = (hwc / 65535.0 * 255.0).round().clip(0, 255).astype(np.uint8)
    elif np.issubdtype(hwc.dtype, np.floating):
        u8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unexpected raster dtype '{hwc.dtype}'")

    pil = Image.fromarray(u8)
    t = resize_transform(pil, patch_size=PATCH_SIZE)
    t_norm = TF.normalize(t, mean=MEAN, std=STD)
    return t_norm

def load_dinov3_backbone(backbone: str, weights_path: str, device: str = "cpu"):
    if backbone == "vitb16":
        model = dinov3_vitb16(pretrained=False)
    elif backbone == "vitl16":
        model = dinov3_vitl16(pretrained=False)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")

    ckpt = torch.load(weights_path, map_location=device)

    # unwrap common checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "teacher" in ckpt:
            state = ckpt["teacher"]
        else:
            state = ckpt
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        clean_state[k[len("module."):] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print("  e.g. missing:", missing[:5])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:5])

    model.eval()
    return model

def extract_embeddings(model, x_bchw: torch.Tensor) -> torch.Tensor:
    """
    Returns (C, Ht, Wt)
    """
    with torch.inference_mode():
        feat = model.get_intermediate_layers(
            x_bchw, n=[len(model.blocks) - 1], reshape=True, norm=True
        )[-1]  # (B,C,Ht,Wt)
        return feat.squeeze(0)

def flatten_tokens(feat_chw: torch.Tensor) -> np.ndarray:
    """
    (C,Ht,Wt) -> (Ht*Wt, C) float32
    """
    C, Ht, Wt = feat_chw.shape
    return (
        feat_chw.permute(1, 2, 0)
        .reshape(Ht * Wt, C)
        .detach().cpu().numpy()
        .astype(np.float32, copy=False)
    )

_hann_cache: Dict[Tuple[int, int], np.ndarray] = {}
def hann2d(h: int, w: int) -> np.ndarray:
    """
    2D Hann weights for blending. Cached by shape.
    """
    key = (h, w)
    if key in _hann_cache:
        return _hann_cache[key]
    wy = np.hanning(h) if h > 1 else np.ones((h,), dtype=np.float32)
    wx = np.hanning(w) if w > 1 else np.ones((w,), dtype=np.float32)
    ww = np.outer(wy, wx).astype(np.float32)
    # avoid all-zeros at tiny sizes
    if np.max(ww) <= 0:
        ww = np.ones((h, w), dtype=np.float32)
    _hann_cache[key] = ww
    return ww

def robust_lo_hi(x: np.ndarray, lo_p=2.0, hi_p=98.0) -> Tuple[float, float]:
    lo = float(np.percentile(x, lo_p))
    hi = float(np.percentile(x, hi_p))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            hi = lo + 1e-6
    return lo, hi

def infer_overlap_blend_possub(
    image_path: str,
    out_path: str,
    pos_template_max_windows: int = 200,     # how many full windows to estimate positional template
    pca_sample_per_window: int = 256,        # how many tokens to sample per window for PCA fit
    pca_max_windows: int = 5000,             # safety cap
    scale_lo_pct: float = 2.0,
    scale_hi_pct: float = 98.0,
):
    weights_path = os.path.join(REPO_DIR, "dinov3", "weights", URL)
    model = load_dinov3_backbone(BACKBONE, weights_path, device=DEVICE)

    rng = np.random.default_rng(0)

    with rasterio.open(image_path) as src:
        print("=== Source image ===")
        print("H,W:", src.height, src.width, "| bands:", src.count, "| res:", src.res)
        print("transform:", src.transform)
        print()

        windows = build_overlapping_windows_with_edges(src, WINDOW_PX, STRIDE_PX)
        print(f"Built {len(windows)} overlapping windows (with edge coverage): win={WINDOW_PX} stride={STRIDE_PX}")

        # Global token grid size (floor to full patches)
        out_h = src.height // PATCH_SIZE
        out_w = src.width // PATCH_SIZE
        if out_h <= 0 or out_w <= 0:
            raise RuntimeError("Image too small for even one patch.")

        # -------------------------------------------------------
        # PASS 0: positional template (full windows only)
        # -------------------------------------------------------
        Ht_full = (WINDOW_PX // PATCH_SIZE)
        Wt_full = (WINDOW_PX // PATCH_SIZE)
        pos_sum: Optional[torch.Tensor] = None
        pos_n = 0

        t0 = time.time()
        for i, win in enumerate(windows, 1):
            # only use full windows for a stable template
            if int(win.height) != WINDOW_PX or int(win.width) != WINDOW_PX:
                continue
            if window_is_empty(src, win):
                continue

            x = window_to_tensor(src, win).unsqueeze(0)   # (1,3,H*,W*) == (1,3,768,768)
            feat = extract_embeddings(model, x)           # (C,48,48) for 768/16
            C, Ht, Wt = feat.shape
            if (Ht, Wt) != (Ht_full, Wt_full):
                continue

            if pos_sum is None:
                pos_sum = torch.zeros_like(feat)
            pos_sum += feat
            pos_n += 1

            if pos_n >= pos_template_max_windows:
                break

            if i % 50 == 0:
                print(f"Pos-template: scanned {i}/{len(windows)} windows; collected {pos_n}")

        if pos_sum is None or pos_n == 0:
            pos_template = None
            print("Pos-template: not built (no valid full non-empty windows).")
        else:
            pos_template = (pos_sum / pos_n).detach()
            print(f"Pos-template built from {pos_n} full windows in {time.time()-t0:.1f}s. shape={tuple(pos_template.shape)}")

        # -------------------------------------------------------
        # PASS 1: PCA fit (sample tokens from many windows)
        # -------------------------------------------------------
        samples = []
        sampled = 0
        kept_windows = 0

        t1 = time.time()
        for i, win in enumerate(windows, 1):
            if kept_windows >= pca_max_windows:
                break
            if window_is_empty(src, win):
                continue

            x = window_to_tensor(src, win).unsqueeze(0)
            feat = extract_embeddings(model, x)  # (C,Ht,Wt)
            C, Ht, Wt = feat.shape

            # subtract positional template if available (top-left slice matches our crop behavior)
            if pos_template is not None:
                # pos_template is (C,Ht_full,Wt_full); take top-left for edge windows
                feat = feat - pos_template[:, :Ht, :Wt]

            tokens = flatten_tokens(feat)  # (N,C)

            # remove per-window global component
            tokens = tokens - tokens.mean(axis=0, keepdims=True)

            take = min(pca_sample_per_window, tokens.shape[0])
            if take > 0:
                idx = rng.choice(tokens.shape[0], size=take, replace=False)
                samples.append(tokens[idx])
                sampled += take
                kept_windows += 1

            if sampled >= TARGET_SAMPLE:
                break

            if i % 50 == 0:
                print(f"PCA-fit: scanned {i}/{len(windows)} | kept {kept_windows} | sampled {sampled}/{TARGET_SAMPLE}")

        if not samples:
            raise RuntimeError("No PCA samples collected (everything empty?).")

        sample_mat = np.concatenate(samples, axis=0)
        print(f"Fitting PCA on {sample_mat.shape[0]} tokens, dim={sample_mat.shape[1]} ...")
        pca = PCA(n_components=N_PC, svd_solver="randomized", whiten=False, random_state=0)
        pca.fit(sample_mat)
        print(f"PCA fitted in {time.time()-t1:.1f}s.")

        # -------------------------------------------------------
        # PASS 2: Overlap + blend into global token grid
        # -------------------------------------------------------
        acc = np.zeros((N_PC, out_h, out_w), dtype=np.float32)
        wgt = np.zeros((out_h, out_w), dtype=np.float32)

        t2 = time.time()
        wrote_windows = 0

        for i, win in enumerate(windows, 1):
            if window_is_empty(src, win):
                continue

            # Map pixel origin -> token origin
            y0_tok = int(win.row_off) // PATCH_SIZE
            x0_tok = int(win.col_off) // PATCH_SIZE

            # Build tensor (cropped to patch multiple) and infer tokens
            x = window_to_tensor(src, win)
            H_star, W_star = int(x.shape[-2]), int(x.shape[-1])
            if H_star <= 0 or W_star <= 0:
                continue

            Ht = H_star // PATCH_SIZE
            Wt = W_star // PATCH_SIZE

            # Clip on the *global* token grid edges (important near bottom/right)
            if y0_tok >= out_h or x0_tok >= out_w:
                continue
            Ht = min(Ht, out_h - y0_tok)
            Wt = min(Wt, out_w - x0_tok)
            if Ht <= 0 or Wt <= 0:
                continue

            feat = extract_embeddings(model, x.unsqueeze(0))  # (C,Ht0,Wt0) where Ht0/Wt0 match H_star/W_star
            C, Ht0, Wt0 = feat.shape

            # Also clip feature maps to the clipped Ht/Wt (if we had to clip to global grid)
            if Ht0 != Ht or Wt0 != Wt:
                feat = feat[:, :Ht, :Wt]
                Ht0, Wt0 = Ht, Wt

            # subtract positional template if available
            if pos_template is not None:
                feat = feat - pos_template[:, :Ht0, :Wt0]

            tokens = flatten_tokens(feat)  # (Ht*Wt,C)
            tokens = tokens - tokens.mean(axis=0, keepdims=True)

            pcs = pca.transform(tokens).astype(np.float32).reshape(Ht0, Wt0, N_PC)  # (Ht,Wt,3)

            # blend weights for this token shape
            blend = hann2d(Ht0, Wt0)

            # accumulate
            for k in range(N_PC):
                acc[k, y0_tok:y0_tok+Ht0, x0_tok:x0_tok+Wt0] += pcs[..., k] * blend
            wgt[y0_tok:y0_tok+Ht0, x0_tok:x0_tok+Wt0] += blend

            wrote_windows += 1
            if wrote_windows % 50 == 0:
                print(f"Blend: wrote {wrote_windows} windows (scanned {i}/{len(windows)})")

        # normalize
        acc /= np.maximum(wgt, 1e-6)
        print(f"PASS2 blended {wrote_windows} windows in {time.time()-t2:.1f}s.")

        # -------------------------------------------------------
        # Visualization scaling (global, robust percentiles)
        # -------------------------------------------------------
        vis = np.zeros_like(acc, dtype=np.uint8)
        for k in range(N_PC):
            # mask out never-touched cells (should be rare with edge coverage)
            valid = wgt > 0
            vals = acc[k][valid] if np.any(valid) else acc[k].ravel()
            lo, hi = robust_lo_hi(vals, lo_p=scale_lo_pct, hi_p=scale_hi_pct)
            band = np.clip((acc[k] - lo) / (hi - lo), 0.0, 1.0)
            vis[k] = (band * 255.0).round().astype(np.uint8)

        # -------------------------------------------------------
        # Write GeoTIFF in token grid space
        # -------------------------------------------------------
        out_transform = src.transform * Affine.scale(PATCH_SIZE, PATCH_SIZE)
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=N_PC,
            height=out_h,
            width=out_w,
            transform=out_transform,
            nodata=None,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for k in range(N_PC):
                dst.write(vis[k], k + 1)

        print("Done. Wrote:", out_path)
        return out_path


# -------------------- RUN --------------------
if __name__ == "__main__":
    infer_overlap_blend_possub(
        "../data/kev_2023-07-21_cropped.tif",
        "../data/kev_2023-07-21_pca_overlap_blend_possub.tif",
        pos_template_max_windows=200,
        pca_sample_per_window=256,
        scale_lo_pct=2.0,
        scale_hi_pct=98.0,
    )
