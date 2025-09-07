# -*- coding: utf-8 -*-
"""
Train_for_gold.py — BioBERT binary classification (NoRelation / HasRelation)

- Unified Causes / AssociatedWith into HasRelation
- Added [E1][/E1][E2][/E2] special tags and resized embeddings
- Optional positive class upsampling (POS_OVERSAMPLE)
- Class weights + Focal Loss (positive class can be boosted)
- Grouping (doc_id / group_id / pmid / file_id)
"""

import os
import random
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
    
    DATA_CSV = r"C:\Users\Administrator\Desktop\Project\RE(3)\re_pairs_gold.csv"

    MAX_LEN = 256
    TRAIN_BATCH = 8
    EVAL_BATCH = 16
    EPOCHS = 8
    LR = 2e-5
    WEIGHT_DECAY = 0.01

    HEAD_TYPES = {"GENE", "VARIANT", "GENE_VARIANT"}
    TAIL_TYPE = "HPO_TERM"

    # Only valid for positive class HasRelation
    POS_OVERSAMPLE = 1.0      # >1 turns on mild upsampling
    USE_FOCAL = True
    FOCAL_GAMMA = 1.5
    POS_BOOST = 1.2           

    GROUP_COL_CANDIDATES = ["doc_id", "group_id", "pmid", "file_id"]


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

        if "text_marked" not in self.df.columns:
            raise ValueError("Requires column text_marked (containing [E1]…[/E1] [E2]…[/E2])")
        if "relation" not in self.df.columns:
            raise ValueError("Column relation is required.")

        self.texts = self.df["text_marked"].astype(str).tolist()
        self.labels = [LABEL2ID[x] for x in self.df["relation"].tolist()]

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
        return item

# ======================= Loss =======================
class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [B]
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)  # [B]
        focal = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

class LossTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None,
                 use_focal: bool = True, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.criterion = FocalLoss(self.class_weights, self.focal_gamma) if self.use_focal \
                         else nn.CrossEntropyLoss(weight=self.class_weights)

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ======================= Main =======================
def main():
    cfg = CFG()
    set_all_seeds(cfg.SEED)

    # 1) load
    if not os.path.exists(cfg.DATA_CSV):
        raise FileNotFoundError(f"找不到数据文件: {cfg.DATA_CSV}")
    df = pd.read_csv(cfg.DATA_CSV)

    # 2) Unify to binary label space + head and tail filtering
    need_cols = ["text_marked", "relation", "e1_type", "e2_type"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少必要列: {c}")

    # Merge old Causes/AssociatedWith into HasRelation
    df["relation"] = df["relation"].replace({"Causes": "HasRelation", "AssociatedWith": "HasRelation"})
    df = df[(df["e1_type"].isin(cfg.HEAD_TYPES)) & (df["e2_type"] == cfg.TAIL_TYPE)].copy()
    df = df[df["relation"].isin(SCHEMA)].copy()
    if len(df) == 0:
        raise ValueError("The amount of data after filtering is 0. Please check the column names and values.")

    # 3) Grouping
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

    # 4) 
    if cfg.POS_OVERSAMPLE > 1.0:
        pos_mask = train_df["relation"] == "HasRelation"
        if pos_mask.any():
            add_n = int(pos_mask.sum() * (cfg.POS_OVERSAMPLE - 1.0))
            upsampled = train_df[pos_mask].sample(n=add_n, replace=True, random_state=cfg.SEED)
            train_df = pd.concat([train_df, upsampled], axis=0, ignore_index=True)
            print("After oversample (train):\n", train_df["relation"].value_counts())

    # 5) tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    added = tokenizer.add_special_tokens(special_tokens)
    has_all = set(["[E1]","[/E1]","[E2]","[/E2]"]).issubset(set(tokenizer.get_vocab().keys()))
    print(f"Added {added} special tokens. Has all specials? {has_all}")

    # 6) datasets
    train_ds = REDataset(train_df, tokenizer, cfg.MAX_LEN)
    dev_ds   = REDataset(dev_df, tokenizer, cfg.MAX_LEN)

    # 7) model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 8) Class weight
    train_labels = np.array([LABEL2ID[x] for x in train_df["relation"]], dtype=np.int64)
    counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    inv = inv / inv.sum() * NUM_LABELS
    inv[1] *= cfg.POS_BOOST  # 1 是 HasRelation
    class_weights = torch.tensor(inv, dtype=torch.float32)
    print("Class weights:", class_weights.tolist())

    # 9) training args
    args = TrainingArguments(
        output_dir="re_runs_bin",
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        per_device_train_batch_size=cfg.TRAIN_BATCH,
        per_device_eval_batch_size=cfg.EVAL_BATCH,
        weight_decay=cfg.WEIGHT_DECAY,
        logging_steps=50,
        logging_dir="re_runs_bin/logs",
        dataloader_pin_memory=False,
    )

    # 10) metrics
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

    # 11) trainer
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

    # 12) train
    trainer.train()

    # 13) evaluate
    try:
        eval_out = trainer.evaluate(eval_dataset=dev_ds)
        print("Eval metrics:", eval_out)
    except Exception as e:
        print("evaluate() Compatibility issues, skip automatic indicators.", e)

    # 14) predict & reports
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

    # 15) save
    save_dir = "re_best_model_bin"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    with open(os.path.join(save_dir, "nr_bias.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({"no_relation_logit_bias": 0.8}, f, ensure_ascii=False, indent=2)
    print(f"\nModel saved to {save_dir}/")

if __name__ == "__main__":
    main()
