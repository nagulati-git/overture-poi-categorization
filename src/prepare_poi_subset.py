import argparse
import json
import os
import urllib.request
import pandas as pd
import geopandas as gpd


DEFAULT_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/OvertureMaps/schema/main/docs/schema/concepts/"
    "by-theme/places/overture_categories.csv"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare POI subset and top-level taxonomy from Overture categories CSV."
    )
    p.add_argument(
        "--places-path",
        default=os.path.join("out_overture_places", "places.geojson"),
        help="Path to places GeoJSON from ingest_places.py",
    )
    p.add_argument(
        "--categories-url",
        default=DEFAULT_CATEGORIES_URL,
        help="Raw URL to overture_categories.csv",
    )
    p.add_argument(
        "--categories-cache",
        default=os.path.join("data", "overture_categories.csv"),
        help="Local cache path for overture_categories.csv",
    )
    p.add_argument("--n", type=int, default=100, help="Number of POIs to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument(
        "--outdir",
        default=os.path.join("data", "week3"),
        help="Output directory",
    )
    return p.parse_args()


def _ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _download_if_needed(url, path):
    if os.path.exists(path):
        return path
    _ensure_parent_dir(path)
    urllib.request.urlretrieve(url, path)
    return path


def _normalize_columns(df):
    return {c.lower().strip(): c for c in df.columns}


def _pick_column(colmap, contains_any, exclude_any=None):
    exclude_any = exclude_any or []
    for key, original in colmap.items():
        if any(token in key for token in contains_any) and not any(
            token in key for token in exclude_any
        ):
            return original
    return None


def _split_path(value):
    if value is None or pd.isna(value):
        return None
    for sep in [" > ", ">", "/", "|"]:
        if sep in str(value):
            return str(value).split(sep)[0].strip()
    return str(value).strip()


def _split_taxonomy_list(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return None
    return text.split(",")[0].strip()


def build_category_to_top_level(df):
    colmap = _normalize_columns(df)
    category_col = _pick_column(colmap, ["category", "value", "name"], exclude_any=["parent", "top"])
    parent_col = _pick_column(colmap, ["parent"])
    top_col = _pick_column(colmap, ["top_level", "toplevel", "top level"])
    path_col = _pick_column(colmap, ["path", "hierarchy", "lineage"])
    taxonomy_col = _pick_column(colmap, ["taxonomy"])

    if category_col is None:
        category_col = df.columns[0]

    if top_col:
        mapping = df[[category_col, top_col]].dropna()
        return dict(zip(mapping[category_col], mapping[top_col]))

    if taxonomy_col:
        mapping = df[[category_col, taxonomy_col]].dropna()
        return {
            row[category_col]: _split_taxonomy_list(row[taxonomy_col])
            for _, row in mapping.iterrows()
        }

    if path_col:
        mapping = df[[category_col, path_col]].dropna()
        return {
            row[category_col]: _split_path(row[path_col]) for _, row in mapping.iterrows()
        }

    if parent_col:
        parent_map = dict(zip(df[category_col], df[parent_col]))

        def root(cat):
            seen = set()
            cur = cat
            while cur in parent_map and cur not in seen and pd.notna(parent_map[cur]):
                seen.add(cur)
                cur = parent_map[cur]
            return cur

        return {cat: root(cat) for cat in parent_map.keys()}

    # Fallback: treat each category as its own top-level
    return {cat: cat for cat in df[category_col].dropna().unique()}


def parse_primary_category(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, dict):
        return raw.get("primary")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data.get("primary")
    return None


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    categories_path = _download_if_needed(args.categories_url, args.categories_cache)
    categories_df = pd.read_csv(categories_path, sep=";")
    category_to_top = build_category_to_top_level(categories_df)

    gdf = gpd.read_file(args.places_path)
    if "categories_json" not in gdf.columns:
        raise SystemExit("places.geojson is missing categories_json column.")

    gdf["overture_primary_category"] = gdf["categories_json"].apply(parse_primary_category)
    gdf["top_level_category"] = gdf["overture_primary_category"].map(category_to_top)

    sample = gdf.sample(n=min(args.n, len(gdf)), random_state=args.seed)

    subset_cols = [
        "id",
        "primary_name",
        "overture_primary_category",
        "top_level_category",
        "confidence",
    ]
    subset_cols = [c for c in subset_cols if c in sample.columns]
    subset = sample[subset_cols]

    subset_path = os.path.join(args.outdir, "poi_subset.csv")
    taxonomy_path = os.path.join(args.outdir, "top_level_categories.csv")
    metrics_path = os.path.join(args.outdir, "baseline_metrics.json")

    subset.to_csv(subset_path, index=False)

    top_level_df = (
        pd.Series(list(category_to_top.values()))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_frame(name="top_level_category")
    )
    top_level_df.to_csv(taxonomy_path, index=False)

    metrics = {
        "total_pois": int(len(gdf)),
        "sample_size": int(len(subset)),
        "with_primary_category": int(gdf["overture_primary_category"].notna().sum()),
        "with_top_level_mapping": int(gdf["top_level_category"].notna().sum()),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Wrote subset:", subset_path)
    print("Wrote taxonomy:", taxonomy_path)
    print("Wrote metrics:", metrics_path)


if __name__ == "__main__":
    main()
