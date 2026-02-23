import os
import json
import time
import argparse
import pandas as pd
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------

MODEL_NAME = "gpt-4.1-mini"
SUBCATEGORY_LIST_PATH = "data/week3/top_level_subcategories_list.csv"
DISAMBIGUATE_TOP_LEVELS = {"arts_and_entertainment", "attractions_and_activities"}

TOP_LEVEL_CATEGORIES = [
    "accommodation",
    "active_life",
    "arts_and_entertainment",
    "attractions_and_activities",
    "automotive",
    "beauty_and_spa",
    "business_to_business",
    "eat_and_drink",
    "education",
    "financial_service",
    "health_and_medical",
    "home_service",
    "mass_media",
    "pets",
    "private_establishments_and_corporates",
    "professional_services",
    "public_service_and_government",
    "real_estate",
    "religious_organization",
    "retail",
    "structure_and_geography",
    "travel",
]

client = OpenAI()


def _load_subcategory_map(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "top_level_category" not in df.columns or "subcategories" not in df.columns:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        top = str(row["top_level_category"]).strip()
        subs = str(row["subcategories"]).strip()
        if top and subs:
            mapping[top] = subs
    return mapping

# -----------------------
# LLM CALL
# -----------------------

def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        # Uncomment if your SDK supports this and you want strict JSON:
        # response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


# -----------------------
# PROMPT
# -----------------------

def _format_optional(label: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return f"{label}: {text}\n"


def build_prompt_from_row(row: pd.Series, subcategory_map: dict) -> str:
    cats = "\n".join(f"- {c}" for c in TOP_LEVEL_CATEGORIES)
    extra = ""
    extra += _format_optional("Basic category", row.get("basic_category"))
    extra += _format_optional("Operating status", row.get("operating_status"))
    extra += _format_optional("Addresses (json)", row.get("addresses_json"))
    extra += _format_optional("Brand (json)", row.get("brand_json"))
    extra += _format_optional("Other names (json)", row.get("names_json"))
    disambig = ""
    if subcategory_map:
        lines = []
        for top in DISAMBIGUATE_TOP_LEVELS:
            if top in subcategory_map:
                lines.append(f"- {top}: {subcategory_map[top]}")
        if lines:
            disambig = "Subcategory lists (for disambiguation):\n" + "\n".join(lines) + "\n"
    return f"""
You are classifying Points of Interest (POIs) into top-level Overture Places categories.

Possible top-level categories:
{cats}
{disambig}

Given the POI details:

Name: "{row.get("primary_name")}"
{extra}

Choose exactly ONE category from the list above.
Return your answer strictly as JSON in this format:

{{"top_level_category": "<one_of_the_categories_above>"}}
"""


# -----------------------
# RESPONSE PARSING
# -----------------------

def parse_response(text: str) -> str:
    text = text.strip()

    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "top_level_category" in data:
            pred = str(data["top_level_category"]).strip()
            if pred in TOP_LEVEL_CATEGORIES:
                return pred
    except json.JSONDecodeError:
        pass

    # Fallback: case-insensitive match
    text_lower = text.lower()
    for cat in TOP_LEVEL_CATEGORIES:
        if cat.lower() in text_lower:
            return cat

    return ""


# -----------------------
# MAIN RUN
# -----------------------

def run(args):
    df = pd.read_csv(args.input_csv)
    subcategory_map = _load_subcategory_map(SUBCATEGORY_LIST_PATH)

    if "primary_name" not in df.columns or "top_level_category" not in df.columns:
        raise SystemExit("poi_subset.csv must have columns 'primary_name' and 'top_level_category'.")

    # Random sample instead of head
    if args.n > 0:
        df = df.sample(n=min(args.n, len(df)), random_state=42)

    preds = []
    errors = 0

    for idx, row in df.iterrows():
        prompt = build_prompt_from_row(row, subcategory_map)

        try:
            resp = call_llm(prompt)
        except Exception as e:
            print(f"[ERROR] Row {idx}: {e}")
            preds.append("")
            errors += 1
            continue

        cat = parse_response(resp)
        preds.append(cat)

        if args.sleep > 0:
            time.sleep(args.sleep)

    df["llm_top_level_category"] = preds

    # -----------------------
    # METRICS
    # -----------------------

    mask_valid = df["llm_top_level_category"] != ""

    accuracy_all = (df["top_level_category"] == df["llm_top_level_category"]).mean()

    accuracy_valid = (
        df.loc[mask_valid, "top_level_category"]
        == df.loc[mask_valid, "llm_top_level_category"]
    ).mean() if mask_valid.sum() else 0.0

    print("\n--- RESULTS ---")
    print(f"Total rows processed: {len(df)}")
    print(f"Valid LLM predictions: {mask_valid.sum()}")
    print(f"Invalid predictions: {len(df) - mask_valid.sum()}")
    print(f"Exact Match Accuracy (all rows): {accuracy_all:.3f}")
    print(f"Exact Match Accuracy (valid only): {accuracy_valid:.3f}")
    print(f"API errors: {errors}")

    # -----------------------
    # SAVE OUTPUTS
    # -----------------------

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv}")

    confusion = pd.crosstab(
        df["top_level_category"],
        df["llm_top_level_category"]
    )
    confusion.to_csv(args.out_confusion)
    print(f"Wrote confusion matrix to {args.out_confusion}")


# -----------------------
# ARGPARSE
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run LLM baseline on poi_subset.csv.")
    p.add_argument(
        "--input-csv",
        default="data/week3/poi_subset.csv",
        help="Path to POI subset CSV.",
    )
    p.add_argument(
        "--out-csv",
        default="data/week3/poi_subset_with_llm.csv",
        help="Where to write POI subset plus LLM predictions.",
    )
    p.add_argument(
        "--out-confusion",
        default="data/week3/llm_top_level_confusion.csv",
        help="Where to write confusion matrix as CSV.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows to process.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between calls (seconds).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
