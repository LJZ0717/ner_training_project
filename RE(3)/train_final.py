# -*- coding: utf-8 -*-
"""
train_final.py —— BioBERT 二分类最终版 (NoRelation / HasRelation)

- 使用全部 gold (+ 可选 silver) 进行最终训练，不再划分 dev
- Tokenizer 增加 [E1][/E1][E2][/E2] 并扩展词表
- 支持样本级权重 (gold=1.0;silver 默认 0.5 或 CSV 里的 weight 列)
- 类权重 + FocalLoss(正类可额外 Boost)
- 训练结束保存到 ./re_best_model_final/，并写入 nr_bias.json (0.8)

目录结构（默认）：
RE(3)/
 ├─ re_pairs_gold.csv
 ├─ re_pairs_silver_kept_bin.csv   # 可无
 └─ train_final.py
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
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
    SILVER_CSV = str((BASE / "re_pairs_silver_kept_bin.csv").resolve())  

    GOLD_WEIGHT   = 1.0
    SILVER_WEIGHT = 0.5   

    MAX_LEN = 256
    TRAIN_BATCH = 8
    EPOCHS = 8
    LR = 2e-5
    WEIGHT_DECAY = 0.01

    
    POS_OVERSAMPLE = 1.0   
    USE_FOCAL   = True
    FOCAL_GAMMA = 1.5
    POS_BOOST   = 1.2      

    HEAD_TYPES = {"GENE", "VARIANT", "GENE_VARIANT"}
    TAIL_TYPE  = "HPO_TERM"

    SAVE_DIR = str((BASE / "re_best_model_final").resolve())

# Label space (binary classification)
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

# ======================= Dataset =======================
class REDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        need = ["text_marked", "relation"]
        for c in need:
            if c not in self.df.columns:
                raise ValueError(f"Required columns {c}。")
        if "sample_weight" not in self.df.columns:
            self.df["sample_weight"] = 1.0

        self.texts   = self.df["text_marked"].astype(str).tolist()
        self.labels  = [LABEL2ID[x] for x in self.df["relation"].tolist()]
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
        ce = self.ce(logits, target)  # [B]
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
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
        sample_weight = inputs.pop("sample_weight", None)  # [B]
        outputs = model(**inputs)
        logits = outputs.logits

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

    # 1) Read gold
    assert os.path.exists(cfg.GOLD_CSV), f"找不到 GOLD_CSV: {cfg.GOLD_CSV}"
    gold = pd.read_csv(cfg.GOLD_CSV)

    need_cols = ["doc_id","sentence","text_marked",
                 "e1_text","e1_type","e1_start","e1_end",
                 "e2_text","e2_type","e2_start","e2_end","relation"]
    gold = gold[need_cols].copy()
    # Unified to two categories
    gold["relation"] = gold["relation"].replace({"Causes":"HasRelation","AssociatedWith":"HasRelation"})
    gold = gold[(gold["e1_type"].isin(cfg.HEAD_TYPES)) & (gold["e2_type"]==cfg.TAIL_TYPE)]
    gold = gold[gold["relation"].isin(SCHEMA)].copy()
    gold["sample_weight"] = cfg.GOLD_WEIGHT

    # 2) silver
    if os.path.exists(cfg.SILVER_CSV):
        silver = pd.read_csv(cfg.SILVER_CSV)
        use_cols = need_cols + [c for c in silver.columns if c in ["weight"]]
        silver = silver[use_cols].copy()
        silver["relation"] = silver["relation"].replace({"Causes":"HasRelation","AssociatedWith":"HasRelation"})
        silver = silver[(silver["e1_type"].isin(cfg.HEAD_TYPES)) & (silver["e2_type"]==cfg.TAIL_TYPE)]
        silver = silver[silver["relation"].isin(SCHEMA)].copy()
        if "weight" in silver.columns:
            silver["sample_weight"] = silver["weight"].astype(float).clip(0.0, 1.0)
        else:
            silver["sample_weight"] = cfg.SILVER_WEIGHT
        df = pd.concat([gold, silver], ignore_index=True)
        print(f"Loaded gold={len(gold)}, silver={len(silver)}, total={len(df)}")
    else:
        df = gold
        print(f"Loaded gold={len(gold)} (no silver)")

    
    if cfg.POS_OVERSAMPLE > 1.0:
        pos_mask = df["relation"] == "HasRelation"
        if pos_mask.any():
            add_n = int(pos_mask.sum() * (cfg.POS_OVERSAMPLE - 1.0))
            df = pd.concat([df, df[pos_mask].sample(n=add_n, replace=True, random_state=cfg.SEED)],
                           ignore_index=True)
            print("After oversample:", df["relation"].value_counts().to_dict())

    print("Label dist (train, full data):")
    print(df["relation"].value_counts())

    # 4) tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})

    # 5) dataset
    train_ds = REDataset(df, tokenizer, cfg.MAX_LEN)

    # 6) model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    model.resize_token_embeddings(len(tokenizer))

    # 7) 类权重
    train_labels = np.array([LABEL2ID[x] for x in df["relation"]], dtype=np.int64)
    counts = np.bincount(train_labels, minlength=NUM_LABELS).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    inv = inv / inv.sum() * NUM_LABELS
    inv[1] *= cfg.POS_BOOST  # 1 = HasRelation
    class_weights = torch.tensor(inv, dtype=torch.float32)
    print("Class weights:", class_weights.tolist())

    # 8) training args
    args = TrainingArguments(
        output_dir=str(Path(cfg.BASE) / "re_runs_bin_final"),
        num_train_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        per_device_train_batch_size=cfg.TRAIN_BATCH,
        weight_decay=cfg.WEIGHT_DECAY,
        logging_steps=50,
        logging_dir=str(Path(cfg.BASE) / "re_runs_bin_final" / "logs"),
        dataloader_pin_memory=False,
    )

    # 9) trainer & train
    trainer = LossTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,
        tokenizer=tokenizer,
        class_weights=class_weights,
        use_focal=cfg.USE_FOCAL,
        focal_gamma=cfg.FOCAL_GAMMA,
    )
    trainer.train()

    # 10) save final
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    
    with open(save_dir / "nr_bias.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"no_relation_logit_bias": 0.8}, f, ensure_ascii=False, indent=2)

    print(f"\nFinal model saved to: {save_dir}")

if __name__ == "__main__":
    main()
