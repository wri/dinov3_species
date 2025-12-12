import rasterio
from rasterio.plot import show
from rasterio.windows import Window

path = "../data/msu_images/cerath_2023-10-01.tif"
cropped_path = "../data/msu_images/cerath_2023-10-01_cropped.tif"
INVALID_VALUES = (0, 255) 
BLOCK = 512             

def find_valid_bbox(src, band_indexes=(1,2,3)):
    rmin = cmin = np.inf
    rmax = cmax = -np.inf
    for _, win in src.block_windows(1):
        arr = src.read(indexes=band_indexes, window=win, masked=False)
        valid = np.any(~np.isin(arr, INVALID_VALUES), axis=0)
        if not valid.any():
            continue
        rows, cols = np.where(valid)
        rows += win.row_off
        cols += win.col_off
        rmin = min(rmin, rows.min())
        rmax = max(rmax, rows.max())
        cmin = min(cmin, cols.min())
        cmax = max(cmax, cols.max())
    if not np.isfinite(rmin):
        return None
    return Window.from_slices((int(rmin), int(rmax) + 1),
                              (int(cmin), int(cmax) + 1))

with rasterio.open(path) as src:
    win = find_valid_bbox(src, (1,2,3))
    if win is None:
        raise ValueError("No valid (non-zero & non-255) pixels found.")

    # Build an output profile with tiling + compression
    profile = src.profile.copy()
    profile.update(
        height=int(win.height),
        width=int(win.width),
        transform=src.window_transform(win),
        tiled=True,
        blockxsize=BLOCK,
        blockysize=BLOCK,
        interleave="pixel",        # better for RGB with compression
        nodata=255,
        compress="zstd",           # or "deflate" or "lzw"
        zlevel=9,                  # for deflate; ignored by zstd
        predictor=2                # horizontal differencing (good for uint8)
    )

    # If the source dtype is larger than needed and values are 0..255, you can shrink:
    cast_to_uint8 = (src.dtypes[0] != "uint8")
    if cast_to_uint8:
        profile.update(dtype="uint8")

    # Create destination and stream data window-by-window to keep memory low
    with rasterio.open(cropped_path, "w", **profile) as dst:
        # iterate over tiles in the *output* space
        for row_off in range(0, dst.height, BLOCK):
            for col_off in range(0, dst.width, BLOCK):
                h = min(BLOCK, dst.height - row_off)
                w = min(BLOCK, dst.width - col_off)
                out_tile = Window(col_off, row_off, w, h)

                # map to source window by offsetting with the crop window origin
                src_tile = Window(win.col_off + col_off, win.row_off + row_off, w, h)

                chunk = src.read(window=src_tile)  # (bands, h, w)

                if cast_to_uint8:
                    # safe cast if your data are truly 0..255; otherwise remove this
                    chunk = chunk.astype("uint8")

                    # ensure nodata stays 255
                    # (not strictly necessary if input already used 255 for nodata)
                    for b in range(chunk.shape[0]):
                        m0 = (chunk[b] == 0)
                        m255 = (chunk[b] == 255)
                        # leave 0 and 255 as-is; everything else already in 1..254

                dst.write(chunk, window=out_tile)

print("Cropped image saved to:", cropped_path)


# Open and read the top-left 512x512 patch
with rasterio.open(path) as src:
    window = Window(col_off=0, row_off=0, width=512, height=512)
    patch = src.read(window=window)  # shape: (3, 512, 512)

# Normalize for display
rgb = patch.astype(float)
#rgb /= rgb.max()

# Visualize
plt.figure(figsize=(8, 8))
plt.imshow(rgb.transpose(1, 2, 0))
plt.title("Top-left 512×512 patch (RGB)")
plt.axis("off")
plt.show()