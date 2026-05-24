"""
Plot experiment results for SEAD project.

This script generates:
1. Main accuracy comparison bar chart.
2. Attention entropy distance comparison bar chart.
3. SEAD beta ablation chart.
4. Tokenizer subword-to-word ratio bar chart.
5. Raw attention entropy comparison bar chart.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_main_results(result_dir):
    df = pd.read_csv(os.path.join(result_dir, "main_results.csv"))

    plt.figure(figsize=(10, 6))
    plt.bar(df["method"], df["accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("Method")
    plt.title("KLUE-NLI Accuracy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 0.65)
    plt.tight_layout()

    output_path = os.path.join(result_dir, "main_results_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_entropy_distance(result_dir):
    df = pd.read_csv(os.path.join(result_dir, "attention_entropy_analysis.csv"))

    student_df = df[df["model"] != "Improved Teacher XLM-R"]

    plt.figure(figsize=(8, 6))
    plt.bar(
        student_df["model"],
        student_df["entropy_distance_to_teacher"]
    )
    plt.ylabel("Entropy Distance to Teacher")
    plt.xlabel("Model")
    plt.title("Attention Entropy Distance to Teacher")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(result_dir, "entropy_distance_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_ablation_results(result_dir):
    df = pd.read_csv(os.path.join(result_dir, "ablation_results.csv"))

    plt.figure(figsize=(8, 6))
    plt.plot(df["beta"].astype(str), df["accuracy"], marker="o")
    plt.ylabel("Accuracy")
    plt.xlabel("Beta")
    plt.title("SEAD Beta Ablation Study")
    plt.ylim(0, 0.65)
    plt.tight_layout()

    output_path = os.path.join(result_dir, "ablation_beta_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_tokenizer_ratio(result_dir):
    df = pd.read_csv(os.path.join(result_dir, "tokenizer_analysis.csv"))

    plt.figure(figsize=(8, 6))
    plt.bar(df["language"], df["subword_word_ratio"])
    plt.ylabel("Subword-to-Word Ratio")
    plt.xlabel("Language")
    plt.title("Tokenizer Fragmentation Analysis")
    plt.tight_layout()

    output_path = os.path.join(result_dir, "tokenizer_ratio_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def plot_attention_entropy_raw(result_dir):
    df = pd.read_csv(os.path.join(result_dir, "attention_entropy_analysis.csv"))

    plt.figure(figsize=(9, 6))
    plt.bar(df["model"], df["attention_entropy"])
    plt.ylabel("Attention Entropy")
    plt.xlabel("Model")
    plt.title("Average Attention Entropy Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(result_dir, "attention_entropy_bar_chart.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")


def main():
    result_dir = "experiments/results"
    os.makedirs(result_dir, exist_ok=True)

    plot_main_results(result_dir)
    plot_entropy_distance(result_dir)
    plot_ablation_results(result_dir)
    plot_tokenizer_ratio(result_dir)
    plot_attention_entropy_raw(result_dir)


if __name__ == "__main__":
    main()
