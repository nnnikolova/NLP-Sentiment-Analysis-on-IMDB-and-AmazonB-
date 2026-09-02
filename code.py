from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, ConcatDataset

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples: dict) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def init_model() -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )


def train_model(model: nn.Module, imdb_dataset: Dataset, amazon_dataset: Dataset) -> nn.Module:
    """
    Each dataset sample contains:
    input_ids: [L]
    attention_mask: [L]
    labels: int (0 or 1)
    """
    # Combine datasets using PyTorch ConcatDataset
    combined_dataset = ConcatDataset([imdb_dataset, amazon_dataset])

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        eval_dataset=amazon_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    return trainer.model
