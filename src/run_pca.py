import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from dinov3.hub.backbones import dinov3_vitl16  # or dinov3_vit7b16 if you prefer
from sklearn.decomposition import PCA

# ------------ USER KNOBS ------------
INPUT_TIF    = "../data/msu_images/cerath_2023-10-01.tif"
SPATIAL_WIN  = 1536                    # non-overlap chip size (px)
READ_MASKED  = False                   # set True to honor nodata
PATCH_SIZE   = 16                      # DINO token stride in pixels
IMAGE_SIZE   = 1536                    # resize target height (px)
MEAN = (0.430, 0.411, 0.296)
STD  = (0.213, 0.156, 0.143)
N_LAYERS = 24                          # number of intermediate layers to request; we use the last one
N_PC     = 3                           # write first 3 PCs
IPCA_BATCH = 20000                     # tokens per partial_fit batch (tune for RAM)
TARGET_SAMPLE = 300_000
RNG = np.random.default.rng(0)

# ------------ Preprocessing (unchanged logic) ------------
def resize_transform(mask_image: Image.Image,
                     image_size: int = IMAGE_SIZE,
                     patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Resize to (H*, W*) where H*, W* are multiples of PATCH_SIZE, then to CHW float [0,1].
    NOTE: preserves aspect ratio of the window; H* is fixed to IMAGE_SIZE.
    """
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)               # e.g., 96
    w_patches = int((w * image_size) / (h * patch_size))   # preserves aspect ratio
    out_h = h_patches * patch_size
    out_w = w_patches * patch_size
    return TF.to_tensor(TF.resize(mask_image, (out_h, out_w)))

def build_windows(src, spatial_win=SPATIAL_WIN):
    """Non-overlapping windows that tile the raster (edge windows are clipped)."""
    h, w = src.height, src.width
    rows = range(0, h, spatial_win)
    cols = range(0, w, spatial_win)
    wins = []
    for y in rows:
        for x in cols:
            win_h = min(spatial_win, h - y)
            win_w = min(spatial_win, w - x)
            if win_h > 0 and win_w > 0:
                wins.append(Window(x, y, win_w, win_h))
    return wins

def window_to_tensor(src, win: Window) -> torch.Tensor:
    """
    Read 3-band window → HWC uint8 → PIL → resize_transform → normalize → CHW float.
    """
    if src.count < 3:
        raise ValueError(f"Expected >=3 bands, found {src.count}")

    arr = src.read([1,2,3], window=win, masked=READ_MASKED)  # (3, H, W)
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(0)
    hwc = np.transpose(arr, (1,2,0))

    # map dtype to uint8 for PIL
    if hwc.dtype == np.uint8:
        u8 = hwc
    elif hwc.dtype == np.uint16:
        u8 = (hwc / 65535.0 * 255.0).round().clip(0,255).astype(np.uint8)
    elif np.issubdtype(hwc.dtype, np.floating):
        u8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    else:
        mn, mx = float(hwc.min()), float(hwc.max())
        u8 = np.zeros_like(hwc, dtype=np.uint8) if mx<=mn else ((hwc-mn)/(mx-mn)*255).round().astype(np.uint8)

    pil = Image.fromarray(u8)  # RGB
    t = resize_transform(pil)            # CHW float32 [0,1]
    t = TF.normalize(t, mean=MEAN, std=STD)
    return t  # CHW

def infer_pipe(image_path: str):

    model = dinov3_vitl16(pretrained=False).eval()

    with rasterio.open(image_path) as src:
        print("=== Source ===")
        print("Size (H, W):", src.height, src.width)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Resolution:", src.res)
        print("Bands:", src.count)
        print()

        # Build windows (non-overlapping)
        windows = build_windows(src, spatial_win=SPATIAL_WIN)
        print(f"Built {len(windows)} non-overlapping windows (tile={SPATIAL_WIN}px)")

        # -------------------------------
        # PASS 1: Sample tokens → fit PCA
        # -------------------------------
        tokens_per_window = max(1, TARGET_SAMPLE // max(1, len(windows)))
        sample_buf = []

        for i, win in enumerate(windows, 1):
            x = window_to_tensor(src, win).unsqueeze(0)  # (1,3,H*,W*)
            with torch.inference_mode():
                feats = model.get_intermediate_layers(x, n=range(N_LAYERS), reshape=True, norm=True)
                f = feats[-1].squeeze(0)                 # (Htok, Wtok, C)
            Ht, Wt, C = f.shape
            tokens = f.reshape(Ht * Wt, C).detach().cpu().numpy().astype(np.float32)  # (Nt, C)

            Nt = tokens.shape[0]
            take = min(Nt, tokens_per_window)
            if take > 0:
                idx = RNG.choice(Nt, size=take, replace=False)
                sample_buf.append(tokens[idx])

            if i % 10 == 0:
                print(f"[sample] processed {i}/{len(windows)} windows")

        if not sample_buf:
            raise RuntimeError("Sampling produced no tokens; check inputs/windows.")

        sample = np.concatenate(sample_buf, axis=0)  # (~TARGET_SAMPLE, C)
        print(f"Fitting PCA on {sample.shape[0]:,} tokens (dim={sample.shape[1]}) ...")

        pca = PCA(n_components=N_PC, svd_solver="randomized", whiten=True, random_state=0)
        pca.fit(sample)
        print("PCA fitted.")

        # -------------------------------------------------------
        # Prepare output GeoTIFF at patch (token) resolution
        # -------------------------------------------------------
        H, W = src.height, src.width
        out_h = math.ceil(H / PATCH_SIZE)
        out_w = math.ceil(W / PATCH_SIZE)

        a, b, c, d, e, f_ = src.transform
        out_transform = Affine(a * PATCH_SIZE, b, c, d, e * PATCH_SIZE, f_)

        profile = src.profile.copy()
        profile.update(
            dtype="uint8",        # visualization product in 0..255
            count=N_PC,           # 3 bands (PC1, PC2, PC3 after viz mapping)
            transform=out_transform,
            width=out_w,
            height=out_h,
            nodata=0,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )

        out_path = os.path.splitext(image_path)[0] + "_pca.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            # optional: initialize bands with zeros
            for b in range(1, N_PC + 1):
                dst.write(np.zeros((out_h, out_w), dtype=np.uint8), b)

            # ------------------------------------------------
            # PASS 2: Transform each window → viz → write
            # ------------------------------------------------
            for i, win in enumerate(windows, 1):
                # location in token grid
                y0_tok = win.row_off // PATCH_SIZE
                x0_tok = win.col_off // PATCH_SIZE

                x = window_to_tensor(src, win).unsqueeze(0)  # (1,3,H*,W*)
                with torch.inference_mode():
                    f = model.get_intermediate_layers(x, n=range(N_LAYERS), reshape=True, norm=True)[-1].squeeze(0)
                Ht, Wt, C = f.shape
                tokens = f.reshape(Ht * Wt, C).detach().cpu().numpy().astype(np.float32)  # (Nt, C)

                # PCs (neg/pos) → your viz mapping: *2 → sigmoid → [0,1] → uint8
                pcs = pca.transform(tokens).astype(np.float32).reshape(Ht, Wt, N_PC)  # (Ht, Wt, 3)
                pcs_t = torch.from_numpy(pcs)                       # H, W, 3
                vis = torch.sigmoid(pcs_t.mul(2.0)).permute(2, 0, 1)  # 3, H, W in [0,1]
                vis_u8 = (vis.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()

                # clip to raster edges on token grid (border windows)
                y1_tok = min(y0_tok + Ht, out_h)
                x1_tok = min(x0_tok + Wt, out_w)
                h_write = y1_tok - y0_tok
                w_write = x1_tok - x0_tok
                if h_write <= 0 or w_write <= 0:
                    continue

                win_tok = Window(x0_tok, y0_tok, w_write, h_write)
                for b in range(N_PC):
                    dst.write(vis_u8[b, :h_write, :w_write], b + 1, window=win_tok)

                if i % 10 == 0:
                    print(f"[write] {i}/{len(windows)} windows -> token win "
                          f"(y={y0_tok}:{y1_tok}, x={x0_tok}:{x1_tok}) size=({Ht},{Wt})")

        print("Done. Wrote:", out_path)
        return out_path
    
# if __name__ == "__main__":
#     main(INPUT_TIF)