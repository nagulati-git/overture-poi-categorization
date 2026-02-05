# Overture Maps: Categorizing POIs Along Overture's Places Category Schema

## Project Overview
This repository contains work for a capstone project exploring the use of large language models to assist with categorizing Places of Interest (POIs) in the Overture Maps dataset according to Overture's Places Category Schema.

The goal is to analyze how automated categorization methods compare to existing labels in terms of consistency, coverage and common error patterns.

## Ingestion Pipeline
The ingestion script pulls data directly from Overture's public S3 release files using DuckDB, filter by bounding box and write outputs for quick inspection.

- `src/ingest_places.py`: downloads **places/POIs** for a configurable bounding box and exports:
  - `places.geojson`
  - `places.png`
  - `places.html`

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install duckdb
```

## Run Ingestion
Run from the project root.

### 1) Ingest Places (recommended starting point)
```bash
python src/ingest_places.py --limit 500
```

Optional arguments:
- `--release`: Overture release version (default: `2026-01-21.0`)
- `--bbox`: `west,south,east,north` (default: `-122.52,37.70,-122.35,37.83`)
- `--outdir`: output folder (default: `out_overture_places`)
- `--limit`: row limit for debugging (`0` means no limit)

## Outputs
After a run, open the generated HTML map in your browser:
- `out_overture_places/places.html`
