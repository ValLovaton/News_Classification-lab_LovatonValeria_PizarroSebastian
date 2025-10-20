# src/train_eval.py
from __future__ import annotations
import os, random, numpy as np
from typing import Dict
from datasets import load_dataset, DatasetDict

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)

def stratified_split_agnews(train=0.70, val=0.15, test=0.15) -> DatasetDict:
   
    assert abs(train + val + test - 1.0) < 1e-8, "Las proporciones deben sumar 1.0"
    raw = load_dataset("ag_news")  # columnas: text, label
    train_val = raw["train"].train_test_split(
        test_size=(1 - train), seed=SEED, stratify_by_column="label"
    )

    test_share = test / (val + test)
    val_test = train_val["test"].train_test_split(
        test_size=test_share, seed=SEED, stratify_by_column="label"
    )
    return DatasetDict(
        train=train_val["train"],
        validation=val_test["train"],
        test=val_test["test"],
    )
