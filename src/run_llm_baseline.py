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

TOP_LEVEL_CATEGORIES = [
    "accommodation",
    "activelife",
    "artsandentertainment",
    "attractionsandactivities",
    "automotive",
    "beautyandspa",
    "businesstobusiness",
    "eatanddrink",
    "education",
    "financialservice",
    "healthandmedical",
    "homeservice",
    "massmedia",
    "pets",
    "privateestablishmentsandcorporates",
    "professionalservices",
    "publicserviceandgovernment",
    "realestate",
    "religiousorganization",
    "retail",
    "structureandgeography",
    "travel",
]

client = OpenAI()

# -----------------------
# LLM CALL
# -----------------------

def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        # comment this out if your SDK errors on response_format:
        # response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


# -----------------------
# PROMPT
# -----------------------

def build_prompt(primaryname: str) -> str:
    cats = "\n".join(f"- {c}" for c in TOP_LEVEL_CATEGORIES)
    return f"""
You are classifying Points of Interest (POIs) into top-level Overture Places categories.

Possible top-level categories:
{cats}

Given the POI name:

\"{primaryname}\"

Choose exactly ONE category from the list above.
Return your answer strictly as JSON in this format:

{{"toplevelcategory": "<one_of_the_categories_above>"}}
"""


# -----------------------
# RESPONSE PARSING
# -----------------------

def parse_response(text: str) -> str:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "toplevelcategory" in data:
            pred = str(data["toplevelcategory"]).strip()
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

    if "primaryname" not in df.columns or "toplevelcategory" not in df.columns:
        raise SystemExit("poi_subset.csv must have columns 'primaryname' and 'toplevelcategory'.")

    # Random sample instead of head
    if args.n > 0:
        df = df.sample(n=min(args.n, len(df)), random_state=42)

    preds = []
    errors = 0

    for idx, row in df.iterrows():
        prompt = build_prompt(str(row["primaryname"]))

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

    df["llm_toplevelcategory"] = preds

    # -----------------------
    # METRICS
    # -----------------------

    mask_valid = df["llm_toplevelcategory"] != ""

    accuracy_all = (df["toplevelcategory"] == df["llm_toplevelcategory"]).mean()

    accuracy_valid = (
        df.loc[mask_valid, "toplevelcategory"]
        == df.loc[mask_valid, "llm_toplevelcategory"]
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
        df["toplevelcategory"],
        df["llm_toplevelcategory"]
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
        default="data/week3/llm_toplevel_confusion.csv",
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
