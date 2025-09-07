# -*- coding: utf-8 -*-
"""
train_for_gold_silver.py — BioBERT binary classification (NoRelation / HasRelation)
- Reads gold and silver and trains them together
- Silver samples can now set sample-level weights
- Tokenizer adds [E1][/E1][E2][/E2]
- Class weights + FocalLoss (instance-level weighting)
- Group splitting (doc_id / group_id / pmid / file_id)
- After training, saves the model to re_best_model_bin_mix/ and writes nr_bias.json (default 0.8)
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ======================= Config =======================
class CFG:
    SEED = 42
    MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

    
    BASE = Path(__file__).resolve().parent
    GOLD_CSV   = str((BASE / "re_pairs_gold.csv").resolve())
    SILVER_CSV = str((BASE / "re_pairs_silver_kept_bin.csv").resolve())   # 没有可设为 None

    
    SILVER_WEIGHT = 0.5
    GOLD_WEIGHT   = 1.0

    MAX_LEN = 256
    TRAIN_BATCH = 8
    EVAL_BATCH  = 16
    EPOCHS = 8
    LR = 2e-5
    WEIGHT_DECAY = 0.01

    
    POS_OVERSAMPLE = 1.0    
    USE_FOCAL   = True
    FOCAL_GAMMA = 1.5
    POS_BOOST   = 1.2       

    HEAD_TYPES = {"GENE", "VARIANT", "GENE_VARIANT"}
    TAIL_TYPE  = "HPO_TERM"

    GROUP_COL_CANDIDATES = ["doc_id", "group_id", "pmid", "file_id"]

# Binary classification labels
SCHEMA = ["NoRelation", "HasRelation"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(SCHEMA)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = 2

# ======================= Utils =======================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_group_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

# ======================= Dataset =======================
class REDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        for col in ["text_marked", "relation"]:
            if col not in self.df.columns:
                raise ValueError(f"Required columns {col}。")
        if "sample_weight" not in self.df.columns:
            self.df["sample_weight"] = 1.0

        self.texts = self.df["text_marked"].astype(str).tolist()
        self.labels = [LABEL2ID[x] for x in self.df["relation"].tolist()]
        self.weights = self.df["sample_weight"].astype(float).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item

# ======================= Loss =======================
class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.class_weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, target):
        # Returns the per-sample loss vector, externally reweighted averaged
        ce = self.ce(logits, target)  # [B]
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        focal = (1.0 - pt) ** self.gamma * ce
        return focal  # [B]

class LossTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None,
                 use_focal: bool = True, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        if self.use_focal:
            self.loss_fn = FocalLoss(self.class_weights, self.focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, reduction="none")

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weight = inputs.pop("sample_weight", None)  # [B] or None
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the sample-by-sample loss
        if self.use_focal:
            loss_vec = self.loss_fn(logits, labels)  # [B]
        else:
            loss_vec = self.loss_fn(logits, labels)  # [B]

        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.to(loss_vec.device)
            loss = (loss_vec * sample_weight).mean()
        else:
            loss = loss_vec.mean()

        return (loss, outputs) if return_outputs else loss

# ======================= Main =======================
def main():
    cfg = CFG()
    set_all_seeds(cfg.SEED)

    # 1) load gold
    assert os.path.exists(cfg.GOLD_CSV), f"GOLD_CSV not found: {cfg.GOLD_CSV}"
    gold = pd.read_csv(cfg.GOLD_CSV)
    need_cols = ["doc_id","sentence","text_marked",
                 "e1_text","e1_type","e1_start","e1_end",
                 "e2_text","e2_type","e2_start","e2_end","relation"]
    gold = gold[need_cols].copy()
    #Unified binary classification
    gold["relation"] = gold["relation"].replace({"Causes":"HasRelation","AssociatedWith":"HasRelation"})
    gold = gold[gold["relation"].isin(SCHEMA)].copy()
    gold["sample_weight"] = cfg.GOLD_WEIGHT

    # 2) optional silver
    df = gold
    if cfg.SILVER_CSV and os.path.exists(cfg.SILVER_CSV):
        silver = pd.read_csv(cfg.SILVER_CSV)
        silver = silver[need_cols + [c for c in silver.columns if c in ["weight"]]].copy()
        silver["relation"] = silver["relation"].replace({"Causes":"HasRelation","AssociatedWith":"HasRelation"})
        silver = silver[silver["relation"].isin(SCHEMA)].copy()
        if "weight" in silver.columns:
            silver["sample_weight"] = silver["weight"].astype(float).clip(lower=0.0, upper=1.0)
        else:
            silver["sample_weight"] = cfg.SILVER_WEIGHT
        df = pd.concat([gold, silver], ignore_index=True)
        print("Loaded silver:", len(silver))
    else:
        print("No silver used.")

    # 3) filter heads/tails & sanity
    df = df[(df["e1_type"].isin(cfg.HEAD_TYPES)) & (df["e2_type"] == cfg.TAIL_TYPE)].copy()
    assert len(df) > 0, "The sample after filtering is 0."

    # 4) split (group-aware if possible)
    group_col = pick_group_col(df, cfg.GROUP_COL_CANDIDATES)
    if group_col is None:
        train_df, dev_df = train_test_split(
            df, test_size=0.2, random_state=cfg.SEED, stratify=df["relation"]
        )
    else:
        groups = df[group_col].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.SEED)
        idx_train, idx_dev = next(gss.split(df, groups=groups))
        train_df = df.iloc[idx_train].copy()
        dev_df   = df.iloc[idx_dev].copy()

    print("Label dist (train):\n", train_df["relation"].value_counts())
    print("Label dist (dev):\n",   dev_df["relation"].value_counts())

    # 5) optional oversample positive
    if cfg.POS_OVERSAMPLE > 1.0:
        pos_mask = train_df["relation"] == "HasRelation"
        if pos_mask.any():
            add_n = int(pos_mask.sum() * (cfg.POS_OVERSAMPLE - 1.0))
            upsampled = train_df[pos_mask].sample(n=add_n, replace=True, random_state=cfg.SEED)
            train_df = pd.concat([train_df, upsampled], ignore_index=True)
            print("After oversample (train):\n", train_df["relation"].value_counts())

    # 6) tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    added = tokenizer.add_special_tokens(special_tokens)

    # 7) datasets
    train_ds = REDataset(train_df, tokenizer, cfg.MAX_LEN)
    dev_ds   = REDataset(dev_df,   tokenizer, cfg.MAX_LEN)

    # 8) model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 9) class weights (inverse training set frequency + positive class Boost)
    train_labels = np.array([LABEL2ID[x] for x in train_df["relation"]], dtype=np.int64)
    counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    inv = inv / inv.sum() * NUM_LABELS
    inv[1] *= cfg.POS_BOOST  # 1 是 HasRelation
    class_weights = torch.tensor(inv, dtype=torch.float32)
    print("Class weights:", class_weights.tolist())

    # 10) training args
    args = TrainingArguments(
        output_dir="re_runs_bin_mix",
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        per_device_train_batch_size=cfg.TRAIN_BATCH,
        per_device_eval_batch_size=cfg.EVAL_BATCH,
        weight_decay=cfg.WEIGHT_DECAY,
        logging_steps=50,
        logging_dir="re_runs_bin_mix/logs",
        dataloader_pin_memory=False,
    )

    # 11) metrics
    def compute_metrics(eval_pred):
        try:
            predictions, labels = eval_pred
        except Exception:
            predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = predictions.argmax(axis=-1)
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall":    recall_score(labels, preds, average="macro", zero_division=0),
            "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        }

    # 12) trainer
    trainer = LossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        use_focal=cfg.USE_FOCAL,
        focal_gamma=cfg.FOCAL_GAMMA,
    )

    # 13) train
    trainer.train()

    # 14) evaluate
    try:
        eval_out = trainer.evaluate(eval_dataset=dev_ds)
        print("Eval metrics:", eval_out)
    except Exception as e:
        print("evaluate() Compatibility issues, skip automatic indicators.", e)

    # 15) predict & reports
    preds = trainer.predict(dev_ds)
    pred_logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
    if isinstance(pred_logits, torch.Tensor):
        pred_logits = pred_logits.detach().cpu().numpy()
    y_true = preds.label_ids
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif y_true is None:
        y_true = np.array([LABEL2ID[x] for x in dev_df["relation"]], dtype=np.int64)

    y_pred = np.asarray(pred_logits).argmax(axis=-1)

    print("\nPredicted label distribution (dev):")
    unique, cnt = np.unique(y_pred, return_counts=True)
    for i, c in zip(unique, cnt):
        print(f"{ID2LABEL[int(i)]}: {int(c)}")

    print("\nClassification report (dev):")
    print(classification_report(
        y_true, y_pred,
        target_names=SCHEMA,
        zero_division=0
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS))))

    # 16) save
    save_dir = "re_best_model_bin_mix"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(Path(save_dir) / "nr_bias.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"no_relation_logit_bias": 0.8}, f, ensure_ascii=False, indent=2)
    print(f"\nModel saved to {save_dir}/")

if __name__ == "__main__":
    main()
