import os
import math
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16  
from sklearn.decomposition import PCA
import time

SPATIAL_WIN  = 768                     # a tile of the image - same as img_size (in px) (was 1536)
READ_MASKED  = False                   # set True to honor nodata
PATCH_SIZE   = 16                      # size of image patches used by DINO (in px)
IMAGE_SIZE   = 768                     # size of each input window fed to model (in px) (was 1536) 
N_PC     = 3                           # write first 3 PCs
TARGET_SAMPLE = 10_000                 # the number of tokens to sample across all windows (sample size for PCA fit)
RNG = np.random.default_rng(0)         # 0 is a seed to ensure PCA gets same sample subset

### vitb16 ###  
MEAN = (0.485, 0.456, 0.406)           # imagenet, use for dinov3_vitb16
STD = (0.229, 0.224, 0.225)            # imagenet, use for dinov3_vitb16
REPO_DIR = "../dinov3"              
BACKBONE = "vitb16"
URL = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

### vitl16 ###
# MEAN = (0.430, 0.411, 0.296)           # satellite, use for dinov3_vitl16 
# STD  = (0.213, 0.156, 0.143)           # satellite, use for dinov3_vitl16
# REPO_DIR = "../dinov3"                
# BACKBONE = "vitl16"
# URL = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


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
    """
    Build non-overlapping windows that tile the raster 
    (edge windows are clipped).
    """
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

def load_dinov3_backbone(backbone: str,
                         weights_path: str,
                         device: str = "cpu"
                         ):
    
    '''
    Initialize model with the provided backbone
    Loads the model weights
    identify structure of sub dicts to properly unwrap the 
    checkpoint and select the right keys to map to the model
    '''

    if backbone == "vitb16":
        model = dinov3_vitb16(pretrained=False)
    elif backbone == "vitl16":
        model = dinov3_vitl16(pretrained=False)

    ckpt = torch.load(weights_path, map_location=device)

    # Pick the right sub-dict if wrapped
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

    # Strip 'module.' prefix from DDP checkpoint
    # this doesn't really apply - no module prefix in dict
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    # 5) Load with diagnostics
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    # if missing is high - might be the wrong checkpt/weights for the backbone
    if missing:
        print("  e.g. missing:", missing[:5])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:5])

    model.eval()
    return model

def extract_embeddings(model, x):
    """
    Returns
        tokens: a 2D array (Ht*Wt, Channels) — one embedding vector per patch.
        Ht, Wt: how many patches tall and wide that window produced.
        Channels: the embedding dimension (e.g. 768).
        Expected order is (48, 48, 768)

    ** NOTE: switching models can change the axis order of the 
    returned feature map **

    n = [len(model.blocks) - 1]   #n = 11
    """
    with torch.inference_mode():
        feat = model.get_intermediate_layers(
            x, n=[len(model.blocks) - 1], reshape=True, norm=True
            )[-1].squeeze(0)                   # (Ht, Wt, C)
    
    # if channels come first in returned feat map, switch them 
    # (channels will always be biggest val)
    if feat.shape[0] > feat.shape[1] and feat.shape[0] > feat.shape[2]:
        feat = feat.permute(1, 2, 0).contiguous()

    Ht, Wt, Channels = feat.shape

    # flatten to (N_tokens, C)
    tokens = (feat.reshape(Ht * Wt, Channels)     
                  .detach().cpu().numpy()
                  .astype(np.float32, copy=False))

    #  debugging
    assert tokens.ndim == 2 and tokens.shape[1] == Channels, f"tokens bad shape: {tokens.shape}"
    return tokens, Ht, Wt, Channels

def infer_pipe(image_path: str, out_path: str):

    '''
    Loads dinov3 model
    Builds windows
    Samples entire image to fit PCA
    Applies
    '''

    # original workflow -- randomly initialized web model (no weights) 
    #model = dinov3_vitb16(pretrained=False).eval()
    
    # attempt 2 -- using John's approach
    # model = dinov3_vitb16(pretrained=False)
    # state_dict = torch.load(URL, map_location="cpu")
    # model.load_state_dict(state_dict, strict=False)

    # attempt 3 - using dinov3 repo documentation
    # WEIGHTS_PATH = os.path.join(REPO_DIR, "dinov3", "weights", URL)

    # model = torch.hub.load(
    #     REPO_DIR,
    #     'dinov3_vitb16',      
    #     source='local',
    #     weights=WEIGHTS_PATH,
    # )
    # model.eval()

    WEIGHTS_PATH = os.path.join(REPO_DIR, "dinov3", "weights", URL)
    model = load_dinov3_backbone(BACKBONE, WEIGHTS_PATH)

    with rasterio.open(image_path) as src:
        print("=== Source ===")
        print("Size (H, W):", src.height, src.width)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Resolution:", src.res)
        print("Bands:", src.count)
        print()

        # Build windows
        windows = build_windows(src, spatial_win=SPATIAL_WIN)
        print(f"Built {len(windows)} non-overlapping windows (size={SPATIAL_WIN}px)")

        # -------------------------------
        # PASS 1: Sample tokens → fit PCA
        # -------------------------------
        # divide target sample evenly across all windows but always take >=1 sample per window
        tokens_per_window = max(1, TARGET_SAMPLE // max(1, len(windows))) 
        sample_buf = []

        start_batch_time = time.time()

        for i, win in enumerate(windows, 1):

            # --- skip windows with all 0.0 values ---
            arr = src.read([1, 2, 3], window=win)
            if not np.any(arr):
                continue

            x = window_to_tensor(src, win).unsqueeze(0)  # (1,3,H*,W*)
            tokens, Ht, Wt, Channels = extract_embeddings(model, x)

            N_tokens = tokens.shape[0] # num tokens extracted from that window 
            take = min(N_tokens, tokens_per_window)
            if take > 0:
                idx = RNG.choice(N_tokens, size=take, replace=False)
                sample_buf.append(tokens[idx])

            if i % 10 == 0:
                elapsed = time.time() - start_batch_time
                avg_per_win = elapsed / 10
                print(f"Processed {i:04d}/{len(windows)} windows "
                    f"→ batch took {elapsed:.1f}s (avg {avg_per_win:.2f}s/window)")
                start_batch_time = time.time()

        if not sample_buf:
            raise RuntimeError("Sampling produced no tokens; check inputs/windows.")
        
        ## DEBUGGING
        for k, a in enumerate(sample_buf):
            if a.ndim != 2:
                raise ValueError(f"sample_buf[{k}] has ndim={a.ndim}, shape={a.shape} (expected 2D (N,C))")
            if k == 0: #look at the first sample to get the expected embedding dim
                expected_embedding_dim = a.shape[1]
            elif a.shape[1] != expected_embedding_dim:
                raise ValueError(
                    f"Feature-dim mismatch at sample_buf[{k}]: "
                    f"cols={a.shape[1]} vs expected {expected_embedding_dim}. "
                    "An unflattened (Ht,Wt[,C]) array was appended."
                )

        sample = np.concatenate(sample_buf, axis=0)  # (~TARGET_SAMPLE, C)
        print(f"Fitting PCA on {sample.shape[0]} tokens...")

        pca = PCA(n_components=N_PC, svd_solver="randomized", whiten=False, random_state=0)
        pca.fit(sample)
        print("PCA fitted.")

        # -------------------------------------------------------
        # Prepare output GeoTIFF 
        # -------------------------------------------------------
        H, W = src.height, src.width
        out_h = math.ceil(H / PATCH_SIZE)
        out_w = math.ceil(W / PATCH_SIZE)

        # increase pixel size by patch size
        out_transform = src.transform * Affine.scale(PATCH_SIZE, PATCH_SIZE)

        profile = src.profile.copy()
        profile.update(
            dtype="uint8",        # visualization product in 0..255
            count=N_PC,           # 3 bands (PC1, PC2, PC3 after viz mapping)
            transform=out_transform,
            width=out_w,
            height=out_h,
            nodata=None,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            # optional: initialize bands with zeros
            for b in range(1, N_PC + 1):
                dst.write(np.zeros((out_h, out_w), dtype=np.uint8), b)

            # ------------------------------------------------
            # PASS 2: Transform each window → viz → write
            # ------------------------------------------------
            for i, win in enumerate(windows, 1):

                # skip windows where every pixel in every band is 0.0
                arr = src.read([1, 2, 3], window=win)
                if not np.any(arr):
                    continue

                # compute location in token grid
                y0_tok = win.row_off // PATCH_SIZE
                x0_tok = win.col_off // PATCH_SIZE

                # get features for the current window
                x = window_to_tensor(src, win).unsqueeze(0)
                tokens, Ht, Wt, Channels = extract_embeddings(model, x)

                # Apply PCAs to compress embeddings
                pcs = pca.transform(tokens).astype(np.float32).reshape(Ht, Wt, N_PC)  # (Ht, Wt, 3)

                # multiply by 2 and pass through sigmoid to convert [0,1] → [0,255] 
                pcs_t = torch.from_numpy(pcs)                                         # H, W, 3
                vis = torch.sigmoid(pcs_t.mul(2.0)).permute(2, 0, 1)                  # 3, H, W 
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
                    print(f"Writing {i}/{len(windows)} windows -> token win "
                          f"(y={y0_tok}:{y1_tok}, x={x0_tok}:{x1_tok}) size=({Ht},{Wt})")

        print("Done. Wrote:", out_path)
        return out_path
    
# if __name__ == "__main__":
#     main(INPUT_TIF)