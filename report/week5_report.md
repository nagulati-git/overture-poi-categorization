# Week 5: Experimental Results and Preliminary Analysis

## Overview

This week we ran the full LLM-based categorization experiment on the 100-POI subset and analyzed the outputs in detail. The goal was to evaluate how well GPT-4.1-mini can assign Overture top-level categories to POIs using only the information available in the Places dataset, and to identify systematic errors that reveal gaps in the taxonomy.

## Experimental Setup

We used the enriched LLM baseline developed in Week 4, which provides the model with the following fields per POI:
- `primary_name`
- `basic_category`
- `operating_status`
- `addresses_json`
- `brand_json`
- `names_json`

To reduce confusion between `arts_and_entertainment` and `attractions_and_activities` — the most commonly confused pair in the initial baseline — we injected the official Overture subcategory lists for those two categories directly into the prompt as disambiguation context. The model was prompted to return exactly one top-level category as a JSON object with temperature set to 0.0 for deterministic outputs.

## Results

The enriched LLM baseline achieved an overall accuracy of **76%** on the 100-POI subset, compared to ~64% with the basic name-only prompt. This represents a 12 percentage point improvement from adding contextual fields and subcategory disambiguation.

### Per-Category Performance

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| attractions_and_activities | 0.89 | 0.94 | 0.91 | 33 |
| professional_services | 0.83 | 0.42 | 0.56 | 12 |
| retail | 1.00 | 0.50 | 0.67 | 10 |
| real_estate | 1.00 | 1.00 | 1.00 | 7 |
| education | 0.75 | 1.00 | 0.86 | 6 |
| eat_and_drink | 0.71 | 1.00 | 0.83 | 5 |
| home_service | 0.50 | 0.50 | 0.50 | 4 |
| active_life | 1.00 | 0.25 | 0.40 | 4 |
| accommodation | 1.00 | 0.33 | 0.50 | 3 |
| health_and_medical | 1.00 | 0.67 | 0.80 | 3 |
| financial_service | 0.67 | 1.00 | 0.80 | 2 |
| public_service_and_government | 0.33 | 0.50 | 0.40 | 2 |
| beauty_and_spa | 0.50 | 1.00 | 0.67 | 2 |
| arts_and_entertainment | 0.50 | 1.00 | 0.67 | 2 |
| mass_media | 1.00 | 1.00 | 1.00 | 1 |
| pets | 0.50 | 1.00 | 0.67 | 1 |
| private_establishments_and_corporates | 0.00 | 0.00 | 0.00 | 1 |
| religious_organization | 0.50 | 1.00 | 0.67 | 1 |
| travel | 1.00 | 1.00 | 1.00 | 1 |

Figure: `data/week5/per_category_f1.png`

## Preliminary Analysis

### Well-Classified Categories
Categories with clear, distinctive real-world semantics performed best. `real_estate`, `mass_media`, and `travel` achieved perfect F1 scores. `attractions_and_activities` — the largest class with 33 examples — achieved F1 of 0.91, a significant improvement over the initial baseline thanks to the subcategory disambiguation context added to the prompt.

### Problematic Categories

**`professional_services` (F1 = 0.56):** Despite high precision, recall is only 0.42, meaning the model misses more than half of true `professional_services` POIs. Many are being predicted as `home_service` or `business_to_business`. This likely reflects a genuine taxonomy ambiguity — Overture's boundary between `professional_services` and `home_service` is not always obvious from POI name and category alone (e.g., a property management company or construction firm).

**`active_life` (F1 = 0.40):** Only 1 out of 4 `active_life` POIs were correctly identified. The model tends to classify these as `attractions_and_activities`, which is semantically reasonable — outdoor trails, boat rentals, and sports facilities sit on the border between the two categories. This suggests the Overture taxonomy may benefit from clearer subcategory definitions distinguishing recreational venues from tourist attractions.

**`accommodation` (F1 = 0.50):** Campgrounds and lodges are frequently predicted as `attractions_and_activities`. This reflects a real labeling challenge: outdoor accommodation sites like campgrounds serve a dual function and their Overture primary categories (e.g., `campground`) do not strongly disambiguate.

**`private_establishments_and_corporates` (F1 = 0.00):** The single example in this category was completely missed. This category is likely too ambiguous without richer contextual signals, and may also be underrepresented in the training distribution of the LLM.

### Taxonomy Quality Implications
The error patterns reveal several areas where Overture's taxonomy creates inherent ambiguity for automated categorization:
1. The boundary between `attractions_and_activities` and `arts_and_entertainment` / `active_life` is semantically blurry and requires subcategory context to resolve.
2. Service-oriented categories (`professional_services`, `home_service`, `business_to_business`) overlap considerably when only name and basic category are available.
3. Multi-purpose venues (campgrounds, community centers) are systematically miscategorized because their primary function is ambiguous from metadata alone.

## Artifacts
- Analysis script: `src/analyze_llm_outputs.py`
- Per-category metrics: `data/week5/per_category_metrics.csv`
- Misclassified examples: `data/week5/error_analysis.csv`
- Top confusion pairs: `data/week5/top_confusion_pairs.csv`
- F1 bar chart: `data/week5/per_category_f1.png`
