"""
Tokenizer analysis script.

This script compares subword fragmentation across English, Chinese,
and Korean datasets under the XLM-R tokenizer.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import os


def count_words(text, language):
    """
    Count words for tokenizer analysis.

    For English and Korean, whitespace splitting is used.
    For Chinese, character-level counting is used as a simple approximation.
    """
    if language == "Chinese":
        return max(len(text.replace(" ", "")), 1)
    else:
        return max(len(text.split()), 1)


def count_subword_tokens(text, tokenizer):
    """
    Count subword tokens produced by the tokenizer.
    """
    return len(tokenizer.tokenize(text))


def analyze_texts(texts, tokenizer, language_name):
    """
    Analyze word count, subword token count, and subword/word ratio.
    """
    records = []

    for text in texts:
        word_count = count_words(text, language_name)
        subword_count = count_subword_tokens(text, tokenizer)
        ratio = subword_count / word_count

        records.append({
            "language": language_name,
            "word_count": word_count,
            "subword_count": subword_count,
            "subword_word_ratio": ratio
        })

    return pd.DataFrame(records)


def main(sample_size=1000):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    print("Loading datasets...")

    xnli_en = load_dataset("facebook/xnli", "en", split="validation")
    xnli_zh = load_dataset("facebook/xnli", "zh", split="validation")
    klue_nli = load_dataset("klue", "nli", split="validation")

    english_texts = [
        item["premise"] + " " + item["hypothesis"]
        for item in xnli_en.select(range(min(sample_size, len(xnli_en))))
    ]

    chinese_texts = [
        item["premise"] + " " + item["hypothesis"]
        for item in xnli_zh.select(range(min(sample_size, len(xnli_zh))))
    ]

    korean_texts = [
        item["premise"] + " " + item["hypothesis"]
        for item in klue_nli.select(range(min(sample_size, len(klue_nli))))
    ]

    print("Analyzing tokenizer fragmentation...")

    en_df = analyze_texts(english_texts, tokenizer, "English")
    zh_df = analyze_texts(chinese_texts, tokenizer, "Chinese")
    ko_df = analyze_texts(korean_texts, tokenizer, "Korean")

    result_df = pd.concat([en_df, zh_df, ko_df], ignore_index=True)

    summary = result_df.groupby("language").mean(numeric_only=True).reset_index()

    os.makedirs("experiments/results", exist_ok=True)

    full_output_path = "experiments/results/tokenizer_analysis_full.csv"
    summary_output_path = "experiments/results/tokenizer_analysis.csv"

    result_df.to_csv(full_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)

    print("\nTokenizer Analysis Summary:")
    print(summary)

    print(f"\nSaved summary to: {summary_output_path}")
    print(f"Saved full results to: {full_output_path}")


if __name__ == "__main__":
    main()
