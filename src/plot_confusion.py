import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser(description="Plot confusion matrix heatmap.")
    p.add_argument(
        "--confusion-csv",
        default="data/week3/llm_top_level_confusion.csv",
        help="Path to confusion matrix CSV from run_llm_baseline.py",
    )
    p.add_argument(
        "--out",
        default="data/week3/llm_top_level_confusion.png",
        help="Output image path",
    )
    p.add_argument(
        "--figsize",
        default="12,10",
        help="Figure size as W,H",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.confusion_csv, index_col=0)

    w, h = [float(x) for x in args.figsize.split(",")]
    plt.figure(figsize=(w, h))
    sns.set_style("whitegrid")
    ax = sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("LLM Predicted")
    ax.set_ylabel("Actual (Overture top-level)")
    ax.set_title("LLM vs Overture Top-level Confusion Matrix")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
