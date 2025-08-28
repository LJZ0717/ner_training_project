# -*- coding: utf-8 -*-
"""
repair_entities_overwrite_v3.py
- 所有输出放到 Extract 路径
- 修复后的基因/HPO 直接覆盖 e1_text/e2_text
"""

import os, re, json
import pandas as pd
from rapidfuzz import fuzz

# ====== 路径配置 ======
BASE      = r"C:\Users\Administrator\Desktop\Project"
IN_PATH   = os.path.join(BASE, r"RE(3)\pred_pairs.csv")
OUT_PATH  = os.path.join(BASE, r"Extract\pred_pairs_entities_repaired.csv")
REPORT    = os.path.join(BASE, r"Extract\repair_report.json")

E1_OPEN, E1_CLOSE = "[E1]", "[/E1]"
E2_OPEN, E2_CLOSE = "[E2]", "[/E2]"
SPACE_RE = re.compile(r"\s+")
STOP_EDGES = {"a","an","the","and","of","in","with","for","to","on","at","by"}

def hpo_from_E1_strict(marked: str) -> str:
    if not isinstance(marked, str) or not marked:
        return ""
    a = marked.find(E1_OPEN); b = marked.find(E1_CLOSE)
    if a == -1 or b == -1 or b <= a: return ""
    inside = marked[a+len(E1_OPEN):b]

    left_stub = ""
    if inside and inside[0].isalnum() and a-1 >= 0 and marked[a-1].isalnum():
        i = a-1
        while i>=0 and marked[i].isalnum(): i-=1
        left_stub = marked[i+1:a]

    right_stub = ""
    if inside and inside[-1].isalnum() and b+len(E1_CLOSE) < len(marked) and marked[b+len(E1_CLOSE)].isalnum():
        j = b+len(E1_CLOSE); k = j
        while k < len(marked) and marked[k].isalnum(): k += 1
        right_stub = marked[j:k]

    s = (left_stub + inside + right_stub).replace("\n"," ")
    s = SPACE_RE.sub(" ", s).strip()

    toks = s.split()
    while toks and toks[0].lower() in STOP_EDGES: toks = toks[1:]
    while toks and toks[-1].lower() in STOP_EDGES: toks = toks[:-1]
    return " ".join(toks)

def gene_from_E2_robust(marked: str) -> str:
    if not isinstance(marked, str) or not marked:
        return ""
    a = marked.find(E2_OPEN); b = marked.find(E2_CLOSE)
    if a == -1 or b == -1 or b <= a: return ""

    inside = marked[a+len(E2_OPEN):b]
    j = b + len(E2_CLOSE)
    post = ""
    while j < len(marked) and marked[j].isupper():
        post += marked[j]; j += 1

    m_tail = re.search(r"([A-Z]+)\s*$", inside)
    if m_tail and post:
        joined = m_tail.group(1) + post
    else:
        joined = ""

    before = marked[max(0, a-15):a]
    after  = marked[b+len(E2_CLOSE): b+len(E2_CLOSE)+15]
    window = before + inside + after

    tokens = re.findall(r"\b[A-Z0-9]{2,10}\b", window)
    if joined:
        tokens.append(joined)

    if tokens:
        if "POLG" in tokens: return "POLG"
        return max(tokens, key=len)
    return inside.strip()

def strip_edges(s: str) -> str:
    s = SPACE_RE.sub(" ", (s or "")).strip()
    toks = s.split()
    while toks and toks[0].lower() in STOP_EDGES: toks = toks[1:]
    while toks and toks[-1].lower() in STOP_EDGES: toks = toks[:-1]
    return " ".join(toks)

def main():
    assert os.path.exists(IN_PATH), f"Not found: {IN_PATH}"
    df = pd.read_csv(IN_PATH, encoding="utf-8", engine="python")

    new_e1, new_e2 = [], []

    for _, r in df.iterrows():
        marked = str(r.get("text_marked","") or "")
        t_e1 = hpo_from_E1_strict(marked)   # HPO
        t_e2 = gene_from_E2_robust(marked)  # Gene

        # 基因清洗：保留大写/数字
        tok = re.search(r"\b[A-Z0-9]{2,10}\b", t_e2 or "")
        if tok:
            gene_val = tok.group(0)
        else:
            tok2 = re.search(r"\b[A-Z0-9]{2,10}\b", marked or "")
            gene_val = tok2.group(0) if tok2 else ""
        hpo_val = strip_edges(t_e1)

        new_e1.append(gene_val)
        new_e2.append(hpo_val)

    # 覆盖原列
    df["e1_text"] = new_e1
    df["e2_text"] = new_e2

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig", quoting=1)
    print(f"[DONE] saved -> {OUT_PATH}  rows={len(df)}")

    ok_gene = int(df["e1_text"].astype(str).str.fullmatch(r"[A-Z0-9]{2,10}").sum())
    bad_gene = len(df) - ok_gene
    report = {
        "rows": len(df),
        "gene_fix_ok": ok_gene,
        "gene_fix_bad": bad_gene,
        "notes": [
            "e1_text 现在应该是标准化后的基因符号（POLG, W748S 等）",
            "e2_text 是清洗后的 HPO 短语"
        ]
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(report)

if __name__ == "__main__":
    main()
