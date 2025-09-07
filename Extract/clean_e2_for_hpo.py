"""
clean_e2_for_hpo.py
- 纯离线清洗 pred_pairs_entities_repaired.csv 的 e2_text
- 处理：去掉开头的叙述动词/连接词；修复少量粘连 token
"""

import os, re
import pandas as pd

BASE = r"C:\Users\Administrator\Desktop\Project"
INP  = os.path.join(BASE, r"Extract\pred_pairs_entities_repaired.csv")
OUT  = os.path.join(BASE, r"Extract\pred_pairs_entities_repaired_clean.csv")

# 1) Initial verb/conjunction (case insensitive)
LEADING_VERB = re.compile(r"^(?:revealed|showed|shows|showing|includes|included)\b\s*(?:that\s+)?(?:a|an|the)?\s*", re.I)

# 2) Common adhesion repair table
GLUE_FIX = [
    (re.compile(r"\bwitha\b", re.I), "with a"),
    (re.compile(r"\bforan\b", re.I), "for an"),
    (re.compile(r"\bofbilateral\b", re.I), "of bilateral"),
    (re.compile(r"\brevealeda\b", re.I), "revealed a"),
    (re.compile(r"\bshoweda\b", re.I), "showed a"),
    (re.compile(r"([A-Za-z])and\b"), r"\1 and"),    # intensityand -> intensity and
    (re.compile(r"\s+and\s*$", re.I), ""),          # 句尾孤立 and 去掉
]

def clean_e2(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()

    # Remove the initial descriptive verb/conjunction (+optional that/article)
    t2 = LEADING_VERB.sub("", t)

    # Repair adhesion
    for pat, rep in GLUE_FIX:
        t2 = pat.sub(rep, t2)

    # Ending: Multiple spaces into one
    t2 = re.sub(r"\s+", " ", t2).strip()
    return t2 if t2 else t  

def main():
    assert os.path.exists(INP), f"Not found: {INP}"
    df = pd.read_csv(INP, encoding="utf-8")

    before = df["e2_text"].copy()
    df["e2_text"] = df["e2_text"].astype(str).apply(clean_e2)

    # 简报
    changed = int((before != df["e2_text"]).sum())
    print(f"[DONE] cleaned e2_text -> {OUT} | changed rows: {changed} / {len(df)}")

    df.to_csv(OUT, index=False, encoding="utf-8-sig", quoting=1)

if __name__ == "__main__":
    main()
