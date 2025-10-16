import os
import glob
from typing import Iterable, Optional, Sequence, List
import geopandas as gpd
import pandas as pd
import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd

def get_msu_projects(src_dir: str = "../data/msu_field/") -> List[str]:
    """
    Scan the msu_field directory and return a sorted, unique list of project keys.
    A 'project key' is everything before the first underscore in the top-level folder name.
    Example: 'ARCOS_RW_XY_Field data_d2' -> 'arcos'
    """
    src_dir = os.path.abspath(src_dir)
    projects = set()

    for path in glob.glob(os.path.join(src_dir, "*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if "_" not in name:
            continue
        key = name.split("_", 1)[0].lower()
        projects.add(key)

    return sorted(projects)


def combine_projects(
    outfile: str,
    crs: Optional[str] = "EPSG:3857",
    src_dir: str = "../data/msu_field/",
    projects: Optional[Sequence[str]] = None,
    recursive: bool = True,
    src_crs_if_missing: Optional[str] = None,
    drop_zero_latlon: bool = True,
    lat_col: str = "Lat_T",
    lon_col: str = "Long_T",
) -> gpd.GeoDataFrame:
    """
    Combine shapefiles from project subfolders under `src_dir` into a single shapefile.

    Parameters
    ----------
    projects : Optional[Sequence[str]]
        If provided, only include projects whose *key* (prefix before first '_') matches these (case-insensitive).
        Example: ['arcos', 'foo'].
    recursive : bool
        If True, search for .shp files recursively inside each project folder.
    src_crs_if_missing : Optional[str]
        If an input shapefile lacks a CRS, set this CRS before any reprojection.
    drop_zero_latlon : bool
        If True and the lat/lon columns exist, drop rows with 0 latitude OR 0 longitude.
    lat_col, lon_col : str
        Names of the latitude/longitude columns (if present) used by the zero filter.
    """
    src_dir = os.path.abspath(src_dir)
    include = set([p.lower() for p in projects]) if projects else None

    # Collect top-level project folders
    project_folders = []
    for entry in os.listdir(src_dir):
        folder_name = os.path.basename(entry)
        folder_path = os.path.join(src_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        if "_" not in folder_name:
            continue

        key = folder_name.split("_", 1)[0].lower()
        if include is not None and key not in include:
            continue

        project_folders.append((key, folder_path))

    combined_frames: List[gpd.GeoDataFrame] = []

    for key, folder_path in project_folders:
        # Find shapefiles in this project folder
        shp_paths = []
        if recursive:
            for root, _, files in os.walk(folder_path):
                shp_paths.extend(os.path.join(root, f) for f in files if f.lower().endswith(".shp"))
        else:
            shp_paths.extend(
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".shp")
            )

        for shp in shp_paths:
            try:
                gdf = gpd.read_file(shp)

                # Handle missing CRS
                if gdf.crs is None and src_crs_if_missing:
                    gdf = gdf.set_crs(src_crs_if_missing)

                # Reproject if requested and possible
                if crs is not None:
                    if gdf.crs is None:
                        # Proceed without reprojection but warn in console
                        print(f"Warning: {shp} has no CRS; cannot reproject to {crs}.")
                    else:
                        gdf = gdf.to_crs(crs)

                # Optional zero lat/lon filter
                if drop_zero_latlon and lat_col in gdf.columns and lon_col in gdf.columns:
                    gdf = gdf[(gdf[lat_col] != 0) & (gdf[lon_col] != 0)]

                gdf["project"] = key

                keep_cols = ['SID','PID','Date','TreeID','Lat_T','Long_T',
                'Species','Cluster','DBH__cm_','Crown_D_Ma',
                'Crown_D_90','Height','Remarks','geometry','project']
                wanted = [c for c in keep_cols if c in gdf.columns] 
                gdf = gdf.loc[:, wanted]
                combined_frames.append(gdf)

            except Exception as e:
                print(f"Error processing shapefile {shp}: {e}")

    if not combined_frames:
        raise ValueError(
            "No shapefiles were found to combine.")

    # Concatenate and set CRS if known
    combined = gpd.GeoDataFrame(pd.concat(combined_frames, ignore_index=True))
    if crs is not None:
        combined = combined.set_crs(crs, allow_override=True)

    combined.columns = combined.columns.str.lower()
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')

    # Ensure output directory exists and write
    outdir = os.path.dirname(os.path.abspath(outfile))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    combined.to_file(outfile)

    return combined


def species_eda(
    gdf: gpd.GeoDataFrame,
    none_tokens: Iterable[str] = ("none",),   # strings to treat as missing (case-insensitive)
    treat_blank_as_na: bool = True,           # drop '' and whitespace-only as missing
    normalize_case: bool = True               # store normalized species as lowercase
):
    """
    EDA + cleaning for the species column.

    Steps:
      1) Count rows where species is NaN or one of none_tokens (case-insensitive), plus optional blanks.
      2) Drop those rows.
      3) Report the number of unique species remaining (case-insensitive).
    Returns cleaned GeoDataFrame and a summary report dict.
    """
    df = gdf.copy()
    nan_count = df['species'].isna().sum()

    # Build a normalized text version for checks
    sp_norm = df['species'].astype(str).str.strip().str.lower()

    # Identify rows that should be treated as missing based on tokens/blank
    none_set = {t.lower() for t in none_tokens}
    is_none_token = sp_norm.isin(none_set)
    is_blank = sp_norm.eq("") if treat_blank_as_na else pd.Series(False, index=df.index)

    # Rows to drop: NaN OR token OR blank
    to_drop_mask = df['species'].isna() | is_none_token | is_blank
    none_count = is_none_token.sum()
    blank_count = is_blank.sum() if treat_blank_as_na else 0
    drop_count = to_drop_mask.sum()
    df = df.loc[~to_drop_mask].copy()

    # Optionally normalize case in the species column for consistency
    if normalize_case:
        df['species'] = df['species'].astype(str).str.strip().str.lower()

    # Count unique species across all remaining rows (case-insensitive via normalization)
    unique_species = df['species'].nunique(dropna=True)

    report = {
        "nan_rows": int(nan_count),
        "none_rows": int(none_count),
        "blank_rows": int(blank_count),
        "dropped_rows_total": int(drop_count),
        "rows_after_clean": int(len(df)),
        "unique_species_count": int(unique_species),

    }

    return df, report

    


def _normalize_species_string(s: str) -> str | None:
    """
    Simplified species-normalization:
      - lowercase
      - unicode normalize (NFKC)
      - strip + collapse whitespace
      - standardize spp/sp tokens
      - drop trailing author initials like 'L.' / 'l.'
      - remove stray leading/trailing punctuation (keep commas for multi-entries)
    """
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if not s:
        return None

    # Unicode normalize (handles odd quotes/spaces)
    s = unicodedata.normalize("NFKC", s)

    # Turn any whitespace runs into single spaces
    s = re.sub(r"\s+", " ", s)

    # Standardize spp/sp tokens
    s = re.sub(r"\bspp?\.?\b", "spp", s)  # 'sp', 'spp.', 'sp.' -> 'spp'

    # Drop trailing author initial like 'L.' or 'l' at end (common in 'psidium guajava l')
    s = re.sub(r"\bl\.?\b$", "", s).strip()

    # Remove leading/trailing punctuation (keep commas to help detect multi-entries)
    s = s.strip(" .;:/|\\\"'`")

    # If it collapsed to empty, treat as None
    return s or None


def normalize_species_column(
    df: pd.DataFrame,
    species_col: str = "species",
    keep_original: bool = True,
    genus_initial_map: dict[str, str] | None = None,   # e.g. {'p': 'persea', 'g': 'grevillea', 'm': 'mangifera'}
    known_fixes: dict[str, str] | None = None          # e.g. {'grevillea robutsa':'grevillea robusta'}
) -> pd.DataFrame:
    """
    Create df['species_norm'] with simplified, documented normalization.
    Optionally:
      - Expand genus initials like 'p. americana' using genus_initial_map.
      - Apply a curated known_fixes dict after normalization.

    Returns a copy so the operation is reversible and easy to audit.
    """
    out = df.copy()
    if keep_original and "species_raw" not in out.columns:
        out["species_raw"] = out[species_col]

    # Base normalization
    out["species_norm"] = out[species_col].map(_normalize_species_string)

    # Optional: expand genus initials (only when pattern looks like 'x. epithet')
    if genus_initial_map:
        pattern = re.compile(r"^([a-z])\.\s*([a-z]+)$")
        def _expand_initial(val):
            if not val:
                return val
            m = pattern.match(val)
            if not m:
                return val
            initial, epithet = m.groups()
            genus = genus_initial_map.get(initial)
            return f"{genus} {epithet}" if genus else val
        out["species_norm"] = out["species_norm"].map(_expand_initial)

    # Optional curated fixes (run AFTER expansions)
    if known_fixes:
        out["species_norm"] = out["species_norm"].replace(known_fixes)

    return out


def suggest_species_merges(
    series: pd.Series,
    min_similarity: float = 0.88,
    by_genus: bool = True,
    min_count: int = 1,
) -> pd.DataFrame:
    """
    Suggest merges for near-duplicate names using difflib similarity.
    - by_genus: only compare names sharing the same first token (safer).
    - min_count: ignore rare forms below this frequency.
    Returns a tidy DataFrame with counts and similarity scores.
    """
    s = series.dropna().astype(str)
    vc = s.value_counts()
    candidates = vc[vc >= min_count].index.tolist()

    def genus_of(x): 
        return x.split()[0] if " " in x else x

    suggestions = []
    if by_genus:
        # Compare within genus buckets
        buckets = {}
        for name in candidates:
            buckets.setdefault(genus_of(name), []).append(name)
        groups = buckets.values()
    else:
        groups = [candidates]

    for group in groups:
        group = sorted(group, key=lambda x: (-vc[x], x))  # frequent first
        for i, a in enumerate(group):
            for b in group[i+1:]:
                sim = SequenceMatcher(None, a, b).ratio()
                if sim >= min_similarity:
                    # Suggest the more frequent spelling as the target
                    target = a if vc[a] >= vc[b] else b
                    source = b if target == a else a
                    suggestions.append({
                        "source": source,
                        "target": target,
                        "similarity": round(sim, 3),
                        "source_count": int(vc[source]),
                        "target_count": int(vc[target]),
                        "genus_bucket": genus_of(a),
                    })

    return pd.DataFrame(suggestions).sort_values(
        ["genus_bucket", "similarity", "target_count", "source_count"],
        ascending=[True, False, False, False],
        ignore_index=True
    )
