"""
Improved Vanilla Knowledge Distillation baseline for KLUE-NLI.

This script trains a student model using soft labels from the improved
fine-tuned XLM-R teacher model.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import evaluate


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
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset


def kd_loss_fn(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    ce_loss = F.cross_entropy(student_logits, labels)

    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    kd_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction="batchmean"
    ) * (temperature ** 2)

    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    return total_loss


def evaluate_model(model, dataloader, device):
    accuracy = evaluate.load("accuracy")
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = outputs.logits
            loss = outputs.loss
            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy.compute(predictions=all_preds, references=all_labels)

    return {
        "eval_loss": total_loss / len(dataloader),
        "eval_accuracy": acc["accuracy"],
    }


def main():
    teacher_model_path = "experiments/teacher_klue_nli_improved/checkpoint-1250"
    student_model_name = "distilbert-base-multilingual-cased"

    result_dir = "experiments/results"
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = load_dataset("klue", "nli")

    train_dataset = dataset["train"].select(range(5000))
    eval_dataset = dataset["validation"].select(range(1000))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    print("Preprocessing dataset...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)

    print("Loading improved teacher model...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_path
    ).to(device)

    print("Loading student model...")
    student_model = AutoModelForSequenceClassification.from_pretrained(
        student_model_name,
        num_labels=3,
    ).to(device)

    teacher_model.eval()
    student_model.train()

    optimizer = AdamW(student_model.parameters(), lr=2e-5)

    num_epochs = 2
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    temperature = 2.0
    alpha = 0.5

    print("Training student model with Improved Vanilla KD...")

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_outputs.logits

            loss = kd_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature,
                alpha=alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average training loss: {avg_train_loss:.4f}")

    print("Evaluating Improved Vanilla KD student model...")
    results = evaluate_model(student_model, eval_loader, device)

    print(results)

    output_path = os.path.join(result_dir, "vanilla_kd_improved_klue_nli_results.txt")
    with open(output_path, "w") as f:
        f.write(str(results))

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
