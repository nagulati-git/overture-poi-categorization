# Week 4 Baseline Analysis and LLM Refinements

## Baseline analysis (existing labels)
We summarized the top-level category distribution for the 100-POI subset and generated a confusion matrix comparing LLM predictions to Overture's top-level labels. The distribution plot highlights class imbalance with a large share of `attractions_and_activities`, and smaller counts in `real_estate`, `eat_and_drink`, and `professional_services`. The confusion matrix shows strongest performance on dominant categories and recurring confusion among business-oriented categories.

Figures:
- `data/week3/top_level_distribution.png`
- `data/week3/llm_top_level_confusion.png`

## LLM baseline refinements
We updated the LLM baseline prompt to incorporate additional context fields from the Places dataset (when available): `basic_category`, `operating_status`, `addresses_json`, `brand_json`, and `names_json`. This improved the LLM baseline accuracy from ~0.64 to ~0.71 on the 100-POI subset.

To address common confusion between `arts_and_entertainment` and `attractions_and_activities`, we also injected the official Overture subcategory lists for those two top-level classes into the prompt. With that disambiguation context, accuracy improved further to ~0.76 on the same subset.

## Artifacts
- Subset and labels: `data/week3/poi_subset.csv`
- LLM predictions: `data/week3/poi_subset_with_llm.csv`
- LLM vs actual comparison: `data/week3/llm_vs_actual.csv`
- Confusion matrix (CSV): `data/week3/llm_top_level_confusion.csv`
- Subcategory lists: `data/week3/top_level_subcategories_list.csv`
