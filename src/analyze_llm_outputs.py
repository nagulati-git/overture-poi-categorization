import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "data/week3/poi_subset_with_llm.csv"
OUT_DIR = "data/week5"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["top_level_category", "llm_top_level_category"])

y_true = df["top_level_category"]
y_pred = df["llm_top_level_category"]

# --- Overall accuracy ---
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc:.2%}")

# --- Per-category metrics ---
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
report_df = report_df[report_df.index.isin(y_true.unique())]
report_df = report_df[["precision", "recall", "f1-score", "support"]].round(2)
report_df.columns = ["Precision", "Recall", "F1", "Support"]
report_df["Support"] = report_df["Support"].astype(int)
report_df = report_df.sort_values("Support", ascending=False)
report_df.to_csv(f"{OUT_DIR}/per_category_metrics.csv")
print(report_df.to_string())

# --- Error analysis: misclassified rows ---
errors_df = df[y_true.values != y_pred.values][
    ["primary_name", "overture_primary_category", "top_level_category", "llm_top_level_category"]
].copy()
errors_df.columns = ["Name", "Primary Category", "True Label", "Predicted Label"]
errors_df.to_csv(f"{OUT_DIR}/error_analysis.csv", index=False)
print(f"\nTotal misclassified: {len(errors_df)}")

# --- Top confusion pairs ---
pairs = (
    errors_df.groupby(["True Label", "Predicted Label"])
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)
pairs.to_csv(f"{OUT_DIR}/top_confusion_pairs.csv", index=False)

# --- F1 bar chart ---
plot_df = report_df[report_df["Support"] > 0].sort_values("F1", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(plot_df.index, plot_df["F1"], color="steelblue")
ax.set_xlabel("F1 Score")
ax.set_title("Per-Category F1 Score (LLM Baseline)")
ax.axvline(x=acc, color="red", linestyle="--", label=f"Overall Accuracy: {acc:.2%}")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_category_f1.png", dpi=150)
plt.close()
print(f"\nOutputs saved to {OUT_DIR}/")
