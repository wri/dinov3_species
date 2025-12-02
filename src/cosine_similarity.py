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
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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


def validate_inputs(trees_path, 
                    image_path, 
                    species_col="species_norm"):
    """
    Validate that:
    1. The tree CRS matches the image CRS (or can be reprojected).
    2. All tree points fall within the image bounds.
    3. The tree dataset contains > 1 species.

    Returns:
        gdf_aligned: GeoDataFrame reprojected to image CRS (safe to use downstream)
        valid_mask: boolean mask of which rows lie inside the image
    """
    gdf = gpd.read_file(trees_path)
    with rasterio.open(image_path) as src_img:
        img_crs = src_img.crs
        img_bounds = src_img.bounds  # (left, bottom, right, top)
        if gdf.crs is None:
            raise ValueError("Tree file has no CRS.")

        if gdf.crs != img_crs:
            print("Reprojecting trees to src_img CRS...")
            gdf = gdf.to_crs(img_crs)

        xmin, ymin, xmax, ymax = img_bounds

        inside_mask = (
            (gdf.geometry.x >= xmin) &
            (gdf.geometry.x <= xmax) &
            (gdf.geometry.y >= ymin) &
            (gdf.geometry.y <= ymax)
        )

        n_outside = (~inside_mask).sum()
        if n_outside > 0:
            print(f"Dropping {n_outside} tree points outside the image extent.")

        gdf_valid = gdf[inside_mask].copy()

        # check species diversity
        unique_species = gdf_valid[species_col].dropna().unique()
        if len(unique_species) < 2:
            raise ValueError(f"Only {len(unique_species)} species present at this site")

        return gdf_valid

def per_tree_features(image_path: str,
                      pca_path: str,
                      trees_path: str,
                      out_path: str = None,
                        ):
    '''
    loads AOI inputs
    # eventually can perform the clip here

    Loads dinov3 model
    Builds windows
    Extract dino embedding for window containing tree point
    Read pca closes to tree point
    Return table with lat, lon, species, token_row, token_col, pca_vec, embed_vec
    '''
    model = dinov3_vitb16(pretrained=False).eval()
    trees = gpd.read_file(trees_path)
    records: list[dict] = []

    trees_valid = validate_inputs(
        trees_path=trees_path,
        image_path=image_path,
        species_col="species_norm",
    )
    print(trees_valid.columns)

    print(f"Extracting information for {len(trees_valid)}/{len(trees)} valid tree points.")
    
    with rasterio.open(image_path) as src_img, rasterio.open(pca_path) as src_pca:
        for idx, row in trees_valid.iterrows():
            pt: Point = row.geometry
            x, y = pt.x, pt.y
            species = row["species_norm"]
            tree_id = row["treeid"] 

            # ------------------------------------------------
            # Get PCA value at tree location
            # ------------------------------------------------
            pca_row, pca_col = rowcol(src_pca.transform, x, y)
            pca_vals = src_pca.read(window=Window(pca_col, pca_row, 1, 1)).squeeze()

            # ensure PCA vector is float32
            pca_vals = np.asarray(pca_vals, dtype=np.float32)

            # ------------------------------------------------
            # Extract DINO embedding on window that contains tree
            # ------------------------------------------------
            # Find the image-space pixel for the point
            row_img, col_img = rowcol(src_img.transform, x, y)
            row_img = int(np.clip(row_img, 0, src_img.height - 1))
            col_img = int(np.clip(col_img, 0, src_img.width  - 1))

            # Choose a window aligned to IMAGE_SIZE that still contains the point.
            half = IMAGE_SIZE // 2

            # top-left corner of the window (aligned to PATCH_SIZE multiples)
            r0 = max(0, (row_img - half) // PATCH_SIZE * PATCH_SIZE)
            c0 = max(0, (col_img - half) // PATCH_SIZE * PATCH_SIZE)

            # bottom-right corner
            r1 = min(src_img.height,  r0 + IMAGE_SIZE)
            c1 = min(src_img.width,   c0 + IMAGE_SIZE)

            # adjust to keep window full-sized where possible
            r0 = max(0, r1 - IMAGE_SIZE)
            c0 = max(0, c1 - IMAGE_SIZE)

            win = Window(c0, r0, c1 - c0, r1 - r0)

            x_tensor = window_to_tensor(src_img, win).unsqueeze(0)

            tokens, Ht, Wt, Channels = extract_embeddings(model, x_tensor)

            # Map tree location to token index (tr, tc)
            r_off = row_img - r0
            c_off = col_img - c0

            # convert to model input coords (scaled to IMAGE_SIZE)
            scale_r = IMAGE_SIZE / (r1 - r0)
            scale_c = IMAGE_SIZE / (c1 - c0)

            r_model = r_off * scale_r
            c_model = c_off * scale_c

            # final token index (patch size = 16)
            tr = int(np.clip(np.floor(r_model / PATCH_SIZE), 0, Ht - 1))
            tc = int(np.clip(np.floor(c_model / PATCH_SIZE), 0, Wt - 1))

            flat = tr * Wt + tc
            embed_vec = tokens[flat]

            # handle torch tensor / numpy array and ensure float32
            if hasattr(embed_vec, "detach"):  # likely a torch tensor
                embed_vec = embed_vec.detach().cpu().numpy()
            embed_vec = np.asarray(embed_vec, dtype=np.float32)

            records.append({
                "x": float(x),
                "y": float(y),
                "species": species,
                "treeid": int(tree_id),
                "r_tok": int(tr),
                "c_tok": int(tc),
                "pca_vec": pca_vals,
                "embed_vec": embed_vec,
            })
    # save as npz
    if out_path is not None and len(records) > 0:
        out_path = Path(out_path)

        xs = np.array([r["x"] for r in records], dtype=np.float64)
        ys = np.array([r["y"] for r in records], dtype=np.float64)
        species_arr = np.array([r["species"] for r in records], dtype=object)
        treeid_arr = np.array([r["treeid"] for r in records], dtype=np.int32)
        r_tok_arr = np.array([r["r_tok"] for r in records], dtype=np.int32)
        c_tok_arr = np.array([r["c_tok"] for r in records], dtype=np.int32)
        pca_mat = np.stack([r["pca_vec"] for r in records]).astype(np.float32, copy=False)
        embed_mat = np.stack([r["embed_vec"] for r in records]).astype(np.float32, copy=False)

        np.savez_compressed(
            out_path,
            x=xs,
            y=ys,
            species=species_arr,
            treeid=treeid_arr,
            r_tok=r_tok_arr,
            c_tok=c_tok_arr,
            pca=pca_mat,
            embed=embed_mat,
        )
        print(f"Saved to {out_path}")

    return records

def calc_similarity_scores(results, 
                           n=None, 
                           verbose=False):

    if n != None:
        subset = results[:n]
    else:
        subset = results

    # Extract species and embeddings
    species = [rec["species"] for rec in subset]
    treeids = [rec["treeid"] for rec in subset] 
    embeddings = np.stack([rec["embed_vec"] for rec in subset])   

    sim_matrix = cosine_similarity(embeddings)   # (5 × 5) matrix
    dist_matrix = 1 - sim_matrix
    species_counts = {sp: species.count(sp) for sp in set(species)}
    print("Species count:", species_counts)
    print("Embeddings shape:", embeddings.shape) # shape (n_species, 768)
    
    if verbose:
        print("\nCosine similarity matrix:")
        print(sim_matrix)
        print("\nCosine distance matrix:")
        print(dist_matrix)
        print("\nPairwise cosine distances:")
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                print(f"Tree {i} ({species[i]}) ↔ Tree {j} ({species[j]}): "
                      f"{dist_matrix[i, j]:.4f}")
                
    return sim_matrix, dist_matrix, species, treeids


def plot_cosine_distance_heatmap(dist_matrix, 
                                 species, 
                                 treeids,
                                 title="Cosine distance heatmap"):
    """
    Plot a heatmap of cosine distances between trees.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square matrix (N x N) of cosine distances (0 = identical, ~1 = orthogonal).
    species : list or array-like of str
        Species labels for each tree, length N.
    title : str, optional
        Title for the plot.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]

    if dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("dist_matrix must be square (N x N).")
    if len(species) != n:
        raise ValueError("len(species) must match dist_matrix size.")
    if len(treeids) != n:
        raise ValueError("len(treeids) must match dist_matrix size.")

    # Labels like "123: SpeciesA"
    labels = [f"{treeid}: {sp}" for treeid, sp in zip(treeids, species)]

    plt.figure(figsize=(13, 12))
    im = plt.imshow(dist_matrix, interpolation="nearest")

    plt.title(title)
    plt.xlabel("treeid / species")
    plt.ylabel("treeid / species")

    plt.xticks(ticks=np.arange(n), labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(n), labels=labels)

    plt.colorbar(im, label="Cosine distance (1 - similarity)")
    plt.tight_layout()
    plt.show()
