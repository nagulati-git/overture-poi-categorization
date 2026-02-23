import argparse
import os
import duckdb
import geopandas as gpd

def parse_args():
    p = argparse.ArgumentParser(description="Download + visualize Overture Places (type=place) for a bbox.")
    p.add_argument("--release", default="2026-01-21.0",
                   help="Overture release version, e.g., 2026-01-21.0")
    p.add_argument("--outdir", default="out_overture_places", help="Output directory.")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional LIMIT for debugging (0 means no limit).")
    p.add_argument("--bbox", default="-122.52,37.70,-122.35,37.83",
                   help="Bounding box west,south,east,north (EPSG:4326).")
    return p.parse_args()

def main():
    args = parse_args()
    west, south, east, north = [float(x) for x in args.bbox.split(",")]

    release = args.release
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Overture S3 path for Places POIs
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"

    out_geojson = os.path.join(outdir, "places.geojson")
    out_png = os.path.join(outdir, "places.png")
    out_html = os.path.join(outdir, "places.html")

    con = duckdb.connect(database=":memory:")

    # Enable cloud reads + spatial functions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")

    limit_sql = f"LIMIT {args.limit}" if args.limit and args.limit > 0 else ""

    # GeoJSON export is picky about nested types.
    # So: convert nested fields to JSON strings using to_json().
    # Note: if your release doesn't have a field, DuckDB will error.
    # If that happens, remove/comment that line and rerun.
    sql_geojson = f"""
    COPY(
      SELECT
        id,
        names.primary AS primary_name,

        -- Optional: may not exist in some releases; remove if DuckDB errors.
        confidence,
        basic_category,
        operating_status,

        -- Often nested; convert to JSON text so export works.
        to_json(names)      AS names_json,
        to_json(sources)    AS sources_json,
        to_json(categories) AS categories_json,
        to_json(addresses)  AS addresses_json,
        to_json(brand)      AS brand_json,

        geometry
      FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
      WHERE
        bbox.xmin <= {east}
        AND bbox.xmax >= {west}
        AND bbox.ymin <= {north}
        AND bbox.ymax >= {south}
        AND geometry IS NOT NULL
      {limit_sql}
    )
    TO '{out_geojson}'
    WITH (FORMAT GDAL, DRIVER 'GeoJSON');
    """

    print("Querying Overture places from:", s3_path)
    print("BBOX:", (west, south, east, north))
    print("Writing GeoJSON:", out_geojson)

    con.execute(sql_geojson)
    con.close()

    # Load & visualize
    gdf = gpd.read_file(out_geojson)

    if gdf.empty:
        raise SystemExit("No places returned for that bbox. Try expanding bbox or check coordinates.")

    # Extract lat/lon for inspection and map tooltips
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x

    # Static plot
    ax = gdf.plot(figsize=(10, 10), markersize=2)
    ax.set_title(f"Overture places (release {release})\nBBOX={west,south,east,north}")
    ax.figure.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote:", out_png)

    # Interactive map
    cols = [c for c in [
        "primary_name",
        "confidence",
        "basic_category",
        "operating_status",
        "names_json",
        "sources_json",
        "categories_json",
        "addresses_json",
        "brand_json",
        "id",
        "lat",
        "lon"
    ] if c in gdf.columns]

    m = gdf.explore(tooltip=cols, popup=cols)
    m.save(out_html)
    print("Wrote:", out_html)

    print("\nDone.")
    print("Open the HTML map:", out_html)
    print("GeoJSON saved at:", out_geojson)

if __name__ == "__main__":
    main()
