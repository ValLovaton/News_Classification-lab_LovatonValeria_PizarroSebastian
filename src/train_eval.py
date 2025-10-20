
# src/train_eval.py
from __future__ import annotations
import os, json, random, numpy as np, torch
from dataclasses import dataclass
from typing import Dict, Any, List
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)

@dataclass
class ModelSpec:
    name: str
    id: str

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("RoBERTa", "roberta-base"),
    ModelSpec("DeBERTa", "microsoft/deberta-v3-base"),
    ModelSpec("ModernBERT", "microsoft/modernbert-base"),
]

def stratified_split_agnews(train=0.70, val=0.15, test=0.15) -> DatasetDict:
    raw = load_dataset("ag_news")
    train_val = raw["train"].train_test_split(test_size=(1-train), seed=SEED, stratify_by_column="label")
    test_share = test / (val + test)
    val_test = train_val["test"].train_test_split(test_size=test_share, seed=SEED, stratify_by_column="label")
    return DatasetDict(train=train_val["train"], validation=val_test["train"], test=val_test["test"])

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # Macro-F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}

def train_and_eval(model_id: str, ds_splits: DatasetDict, outdir: str) -> Dict[str, Any]:
    num_labels = 4
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenized = ds_splits.map(lambda b: tokenize_function(b, tokenizer), batched=True)
    cols = [c for c in tokenized["train"].column_names if c not in ("input_ids","attention_mask","label")]
    tokenized = tokenized.remove_columns(cols)
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=outdir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"])
    return {"test_metrics": test_metrics}
