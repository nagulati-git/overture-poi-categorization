# Overture POI Categorization

This project evaluates how well an LLM can classify Overture Maps points of interest (POIs) into Overture's top-level Places taxonomy. The current workflow ingests Overture Places data for a bounding box, maps each POI's primary category to a top-level label, runs a GPT-based baseline, and analyzes classification errors.

## Repository scope

The repository currently includes:

- An ingestion script that pulls Overture Places data from the public S3 release.
- A preparation step that samples POIs and derives top-level labels from Overture's category taxonomy.
- An LLM baseline that predicts one of 22 top-level Overture categories.
- Analysis scripts and saved artifacts for Week 3 to Week 5 project milestones.

## Project structure

- `src/ingest_places.py`: download Places data for a bounding box and export map-ready files.
- `src/prepare_poi_subset.py`: build a sampled POI subset and top-level category mapping.
- `src/run_llm_baseline.py`: run the GPT baseline on the sampled POIs.
- `src/plot_confusion.py`: render a confusion-matrix heatmap from baseline output.
- `src/analyze_llm_outputs.py`: compute accuracy, per-category metrics, and error-analysis artifacts.
- `data/week3/`: subset, taxonomy, confusion matrix, and baseline artifacts.
- `data/week5/`: per-category metrics, error analysis, and F1 plot.
- `report/`: weekly writeups summarizing findings.
- `out_overture_places/`: exported map and raw geographic outputs from ingestion.

## Setup

Create a virtual environment, activate it, and install dependencies:

```bash
pip install -r requirements.txt
pip install duckdb openai scikit-learn
```

To run the LLM baseline, set your OpenAI API key:

```bash
set OPENAI_API_KEY=your_key_here
```

On macOS or Linux, use `export OPENAI_API_KEY=your_key_here` instead.

## End-to-end workflow

Run commands from the project root.

### 1. Ingest Overture Places data

```bash
python src/ingest_places.py --limit 500
```

Default settings:

- Release: `2026-01-21.0`
- Bounding box: `-122.52,37.70,-122.35,37.83` (San Francisco area)
- Output directory: `out_overture_places`

Main outputs:

- `out_overture_places/places.geojson`
- `out_overture_places/places.png`
- `out_overture_places/places.html`

Useful options:

- `--release`: Overture release version
- `--bbox`: `west,south,east,north`
- `--outdir`: output directory
- `--limit`: row limit for debugging; `0` means no limit

### 2. Prepare the POI subset and taxonomy

```bash
python src/prepare_poi_subset.py --n 100 --seed 42
```

This step:

- Loads `out_overture_places/places.geojson`
- Caches `data/overture_categories.csv` if needed
- Extracts each POI's Overture primary category
- Maps that category to a top-level Overture label
- Samples a subset for evaluation

Outputs:

- `data/week3/poi_subset.csv`
- `data/week3/top_level_categories.csv`
- `data/week3/baseline_metrics.json`

Useful options:

- `--places-path`: input GeoJSON path
- `--categories-url`: source taxonomy CSV URL
- `--categories-cache`: local taxonomy cache
- `--n`: sample size
- `--seed`: random seed
- `--outdir`: output directory

### 3. Run the LLM baseline

```bash
python src/run_llm_baseline.py --n 100
```

The baseline uses `gpt-4.1-mini` and prompts the model to return exactly one top-level Overture category as JSON. The prompt includes:

- `primary_name`
- `basic_category`
- `operating_status`
- `addresses_json`
- `brand_json`
- `names_json`

It also injects subcategory lists for:

- `arts_and_entertainment`
- `attractions_and_activities`

Outputs:

- `data/week3/poi_subset_with_llm.csv`
- `data/week3/llm_top_level_confusion.csv`

Useful options:

- `--input-csv`: input POI subset
- `--out-csv`: predictions output path
- `--out-confusion`: confusion matrix CSV path
- `--n`: number of rows to process
- `--sleep`: delay between API calls

### 4. Plot the confusion matrix

```bash
python src/plot_confusion.py
```

Output:

- `data/week3/llm_top_level_confusion.png`

### 5. Analyze results

```bash
python src/analyze_llm_outputs.py
```

Outputs:

- `data/week5/per_category_metrics.csv`
- `data/week5/error_analysis.csv`
- `data/week5/top_confusion_pairs.csv`
- `data/week5/per_category_f1.png`


## Notes and limitations

- The ingestion step reads directly from Overture's public S3 release through DuckDB.
- `src/prepare_poi_subset.py` may download `overture_categories.csv` if it is not already cached locally.
- The repository contains generated artifacts from prior runs, so you can inspect outputs without rerunning the full pipeline.
- Evaluation is currently at the top-level category only, not full subcategory classification.

## Related artifacts

- Weekly summaries: `report/week3.md`, `report/week4.md`, `report/week5_report.md`
- Cached taxonomy: `data/overture_categories.csv`
- Example exported map: `out_overture_places/places.html`
