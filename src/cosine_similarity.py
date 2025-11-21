from pathlib import Path
import numpy as np
import math
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
from run_pca import build_windows, window_to_tensor, extract_embeddings
import torch
import torchvision.transforms.functional as TF
from dinov3.hub.backbones import dinov3_vitb16
from rasterio.windows import Window
from rasterio.transform import rowcol
from shapely.geometry import Point

IMAGE_SIZE = 768
PATCH_SIZE = 16
MEAN = (0.430, 0.411, 0.296)
STD  = (0.213, 0.156, 0.143)

def build_tree_aoi(image_path: str,
                   trees_path: str,
                   buffer_m: float | None = 20.0,
):
    """
    Build an AOI geometry around tree points, in the CRS of `image_path`.
    """
    with rasterio.open(image_path) as src_img:
        img_crs = src_img.crs
        gdf = gpd.read_file(trees_path)
        print("Tree CRS:", gdf.crs, " | Image CRS:", img_crs)

        if gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)

        # Buffer distance in this CRS
        if img_crs.is_geographic:
            # crude: meters → degrees at equator
            buf = (buffer_m or 0) / 111_320.0
        else:
            buf = buffer_m or 0.0

        # Union + dissolve all buffers into one polygon
        buffered = [geom.buffer(buf) for geom in gdf.geometry if geom is not None]
        aoi_geom = unary_union(buffered)

        # Clip AOI to image extent so we don't request off-image pixels
        img_poly = box(*src_img.bounds)
        aoi_geom_img = aoi_geom.intersection(img_poly)

    return aoi_geom_img, img_crs


def clip_raster_with_aoi(
    raster_path: str,
    aoi_geom,
    aoi_crs,
    out_path: str,
):
    """
    Clip a raster to an AOI geometry and save the result.

    Parameters
    ----------
    raster_path : str
        Path to source raster.
    aoi_geom : shapely geometry
        AOI geometry in CRS `aoi_crs`.
    aoi_crs : rasterio.crs.CRS or pyproj.CRS
        CRS of `aoi_geom`.
    out_path : str
        Output path for clipped raster.
    """

    with rasterio.open(raster_path) as src:
        # Reproject AOI if needed
        if src.crs is not None and src.crs != aoi_crs:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom], crs=aoi_crs)
            aoi_gdf = aoi_gdf.to_crs(src.crs)
            aoi_geom_src = aoi_gdf.geometry.iloc[0]
        else:
            aoi_geom_src = aoi_geom

        # Clip AOI to raster bounds
        raster_poly = box(*src.bounds)
        aoi_geom_src = aoi_geom_src.intersection(raster_poly)

        shapes = [aoi_geom_src.__geo_interface__]

        out_img, out_transform = mask(src, shapes=shapes, crop=True)

        profile = src.profile
        profile.update(
            height=out_img.shape[1],
            width=out_img.shape[2],
            transform=out_transform,
        )

        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_img)

    print(f"Wrote AOI clip for {raster_path} to {out_path}")


def aoi_clip(
    image_path: str,
    pca_path: str,
    trees_path: str,
    img_out_path: str,
    pca_out_path: str,
    buffer_m: float | None = 20.0,
):
    """
    Export AOI clips of `image_path` and `pca_path` that tightly cover all
    tree points using a user-defined buffer.
    """

    aoi_geom_img, img_crs = build_tree_aoi(
        image_path=image_path,
        trees_path=trees_path,
        buffer_m=buffer_m,
    )

    clip_raster_with_aoi(
        raster_path=image_path,
        aoi_geom=aoi_geom_img,
        aoi_crs=img_crs,
        out_path=img_out_path,
    )

    clip_raster_with_aoi(
        raster_path=pca_path,
        aoi_geom=aoi_geom_img,
        aoi_crs=img_crs,
        out_path=pca_out_path,
    )


def per_tree_features(image_path: str,
                      pca_path: str,
                      trees_path: str,
                        ):
    '''
    loads AOI inputs
    # eventually can perform the clip here

    Loads dinov3 model
    Builds windows
    Extract dino embedding for window containing tree point
    Read pca closes to tree point
    Create table with lat, lon, species, token_row, token_col, pca_vec, embed_vec
    Return table
    '''
    model = dinov3_vitb16(pretrained=False).eval()
    trees = gpd.read_file(trees_path)
    records = []
    print(f"Extracting information for {trees.shape[0]} tree points.")
    with rasterio.open(image_path) as src_img, rasterio.open(pca_path) as src_pca:
        for idx, row in trees.iterrows():
            pt: Point = row.geometry
            x, y = pt.x, pt.y
            species = row['species_norm']

            # ------------------------------------------------
            # Get PCA value at tree location
            # ------------------------------------------------
            print("Extracting PCA")
            pca_row, pca_col = rowcol(src_pca.transform, x, y)
            pca_vals = src_pca.read(window=Window(pca_col, pca_row, 1, 1)).squeeze()

            # ------------------------------------------------
            # Extract dino embedding on window that contains tree
            # ------------------------------------------------
            
            # Find the image-space pixel for the point
            # get rows and cols coordinates given geo coordinates
            row_img, col_img = rowcol(src_img.transform, x, y)  
            row_img = int(np.clip(row_img, 0, src_img.height - 1))
            col_img = int(np.clip(col_img, 0, src_img.width  - 1))

            # Choose a window aligned to IMAGE_SIZE that still contains the point.
            # We align window top-left to a multiple of PATCH_SIZE for clean token grid alignment.
            half = IMAGE_SIZE // 2
            r0 = max(0, (row_img - half) // PATCH_SIZE * PATCH_SIZE)
            c0 = max(0, (col_img - half) // PATCH_SIZE * PATCH_SIZE)
            r1 = min(src_img.height,  r0 + IMAGE_SIZE)
            c1 = min(src_img.width,   c0 + IMAGE_SIZE)

            # Adjust if near image edges
            r0 = max(0, r1 - IMAGE_SIZE)
            c0 = max(0, c1 - IMAGE_SIZE)

            win = Window(c0, r0, c1 - c0, r1 - r0)

            print("Extracting embeddings")
            x_tensor = window_to_tensor(src_img, win).unsqueeze(0)
            # tokens → (Ht*Wt, C)
            tokens, Ht, Wt, Channels = extract_embeddings(model, x_tensor)
     
            # Map tree location to token index which will be tr, tc
            r_off = row_img - r0
            c_off = col_img - c0

            # convert to model input coords (scaled to 768)
            scale_r = IMAGE_SIZE / (r1 - r0)
            scale_c = IMAGE_SIZE / (c1 - c0)

            r_model = r_off * scale_r
            c_model = c_off * scale_c

            # final token index (patch size = 16)
            tr = int(np.clip(np.floor(r_model / PATCH_SIZE), 0, Ht - 1))
            tc = int(np.clip(np.floor(c_model / PATCH_SIZE), 0, Wt - 1))

            # flatten idx
            flat = tr * Wt + tc

            emb_vec = tokens[flat]

            # build the table
            records.append({
                "x": x,
                "y": y,
                "species": species,
                "r_tok": int(tr),
                "c_tok": int(tc),
                "pca_vec": pca_vals.astype(np.float32),
                "embed_vec": emb_vec,
            })
    return records

