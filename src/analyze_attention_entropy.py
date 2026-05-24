"""
Attention entropy analysis for Teacher, Vanilla KD student, and SEAD student.

This script computes the average attention entropy on KLUE-NLI validation samples.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def preprocess_dataset(dataset, tokenizer, max_length=128):
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(preprocess_function, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    return dataset


def compute_attention_entropy(attentions, attention_mask=None, eps=1e-12):
    entropy_values = []

    for layer_attention in attentions:
        attention_probs = layer_attention.clamp(min=eps)

        entropy = -torch.sum(
            attention_probs * torch.log(attention_probs),
            dim=-1
        )

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            entropy = entropy * mask
            entropy = entropy.sum() / mask.sum().clamp(min=1)
        else:
            entropy = entropy.mean()

        entropy_values.append(entropy)

    return torch.stack(entropy_values).mean().item()


def analyze_model(model_name, model_path, tokenizer_name, dataset, device, batch_size=8):
    print(f"Analyzing {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_attentions=True
    ).to(device)

    model.eval()

    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

    entropy_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=model_name):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

            entropy = compute_attention_entropy(
                outputs.attentions,
                attention_mask=attention_mask
            )

            entropy_scores.append(entropy)

    avg_entropy = sum(entropy_scores) / len(entropy_scores)

    print(f"{model_name} average attention entropy: {avg_entropy:.4f}")

    return avg_entropy


def main():
    result_dir = "experiments/results"
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading KLUE-NLI validation dataset...")
    dataset = load_dataset("klue", "nli")["validation"].select(range(1000))

    teacher_path = "experiments/teacher_klue_nli_improved/checkpoint-1250"
    vanilla_kd_student_path = "experiments/vanilla_kd_improved_student"
    sead_student_path = "experiments/sead_beta0001_student"

    teacher_tokenizer = "xlm-roberta-base"
    student_tokenizer = "distilbert-base-multilingual-cased"

    teacher_entropy = analyze_model(
        model_name="Improved Teacher XLM-R",
        model_path=teacher_path,
        tokenizer_name=teacher_tokenizer,
        dataset=dataset,
        device=device,
        batch_size=8,
    )

    vanilla_kd_entropy = analyze_model(
        model_name="Improved Vanilla KD Student",
        model_path=vanilla_kd_student_path,
        tokenizer_name=student_tokenizer,
        dataset=dataset,
        device=device,
        batch_size=8,
    )

    sead_entropy = analyze_model(
        model_name="SEAD beta=0.001 Student",
        model_path=sead_student_path,
        tokenizer_name=student_tokenizer,
        dataset=dataset,
        device=device,
        batch_size=8,
    )

    results = [
        {
            "model": "Improved Teacher XLM-R",
            "accuracy": 0.499,
            "attention_entropy": teacher_entropy,
        },
        {
            "model": "Improved Vanilla KD Student",
            "accuracy": 0.547,
            "attention_entropy": vanilla_kd_entropy,
        },
        {
            "model": "SEAD beta=0.001 Student",
            "accuracy": 0.544,
            "attention_entropy": sead_entropy,
        },
    ]

    df = pd.DataFrame(results)

    df["entropy_distance_to_teacher"] = (
        df["attention_entropy"] - teacher_entropy
    ).abs()

    output_path = os.path.join(result_dir, "attention_entropy_analysis.csv")
    df.to_csv(output_path, index=False)

    print("\nAttention Entropy Analysis Results:")
    print(df)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
