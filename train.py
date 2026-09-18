"""
train.py
--------
Fine-tunes a DistilBERT model on the Jigsaw Toxic Comment Classification
dataset (Kaggle) to classify text as TOXIC (1) or SAFE (0).

Dataset:
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
  Download `train.csv` and place it in `data/train.csv` before running.
  The file has columns: id, comment_text, toxic, severe_toxic, obscene,
  threat, insult, identity_hate. This script collapses the six toxicity
  sub-labels into a single binary label: toxic = 1 if ANY of the six
  columns is 1, else 0.

Usage:
  python train.py \
      --data_path data/train.csv \
      --output_dir model_artifacts \
      --sample_size 15000 \
      --epochs 3

Output:
  A fine-tuned model + tokenizer saved to `model_artifacts/`, ready to be
  loaded by `model.py` / `main.py`.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

MODEL_NAME = "distilbert-base-uncased"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LENGTH = 512


def load_and_prepare_data(csv_path: str, sample_size: int | None) -> pd.DataFrame:
    """Load the Jigsaw CSV and collapse the 6 sub-labels into 1 binary label."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["comment_text"])
    df["label"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    df = df[["comment_text", "label"]].rename(columns={"comment_text": "text"})

    if sample_size is not None and sample_size < len(df):
        # Stratified sample so the toxic/safe ratio is preserved.
        toxic = df[df["label"] == 1]
        safe = df[df["label"] == 0]
        toxic_frac = len(toxic) / len(df)
        n_toxic = int(sample_size * toxic_frac)
        n_safe = sample_size - n_toxic
        df = pd.concat(
            [
                toxic.sample(n=min(n_toxic, len(toxic)), random_state=42),
                safe.sample(n=min(n_safe, len(safe)), random_state=42),
            ]
        ).sample(frac=1, random_state=42)  # shuffle

    return df.reset_index(drop=True)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--output_dir", type=str, default="model_artifacts")
    parser.add_argument("--sample_size", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    # Precision is the priority metric here (target: 91%+), so the decision
    # threshold on P(toxic) can be raised above the default 0.5 to trade a
    # little recall for higher precision if needed. Adjust after evaluating.
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Could not find {args.data_path}. Download the Jigsaw Toxic "
            "Comment dataset from Kaggle and place train.csv there. See "
            "the module docstring in train.py for the link."
        )

    print(f"Loading data from {args.data_path} ...")
    df = load_and_prepare_data(args.data_path, args.sample_size)
    print(f"Loaded {len(df)} rows | toxic={df['label'].sum()} | safe={(df['label']==0).sum()}")

    train_df, eval_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    eval_ds = eval_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    eval_ds = eval_ds.rename_column("label", "labels")
    columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "safe", 1: "toxic"}, label2id={"safe": 0, "toxic": 1}
    )

    training_args = TrainingArguments(
        output_dir="./train_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting fine-tuning ...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    if metrics.get("eval_precision", 0) < 0.91:
        print(
            "\nWARNING: precision is below the 91% target. Consider:\n"
            "  - raising --decision_threshold in main.py's inference logic\n"
            "  - training more epochs / more data\n"
            "  - class-weighting the loss (toxic comments are a minority class)\n"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
