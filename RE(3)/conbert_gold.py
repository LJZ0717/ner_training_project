# -*- coding: utf-8 -*-
"""
convert_gold.py  —— 二分类版本 (NoRelation / HasRelation)

从多文件 JSONL(每行: {"text": "...", "spans":[...]}) 中抽取 (head, tail) 对，
基于句子/触发词产出关系，并写出 re_pairs_gold.csv。

主要变化：
- 统一输出为二分类:NoRelation / HasRelation
- 同句命中【因果/关联】触发词 → HasRelation;仅“检测/提到” → NoRelation
- 相邻句(±WINDOW_SENT)若有因果/关联触发词 → HasRelation;否则可记为负例
- 远距对不过度造负例（跳过）
- 负例按 NEG_RATIO 相对正例下采样
"""

import os, json, re, glob, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import pandas as pd

# ========= 0) CONFIG =========
INPUT_DIR   = r"C:\Users\Administrator\Desktop\Project\Resource\annotations_new"   # gold jsonl 文件夹
OUTPUT_CSV  = r"C:\Users\Administrator\Desktop\Project\RE(3)\re_pairs_gold.csv"
SEED        = 42
NEG_RATIO   = 0.8     # 每个正样本最多保留 1.5 个 NoRelation
WINDOW_SENT = 1        # 允许“相邻句”窗口大小（1=同句或相邻句）

# 允许的实体类型
HEAD_TYPES  = {"GENE", "VARIANT", "GENE_VARIANT"}
TAIL_TYPES  = {"HPO_TERM"}

# 触发词（全部小写）
CAUSES_TRIGGERS = [
    r"caused by", r"due to", r"result(?:ed)? from", r"attribut(?:e|ed) to",
    r"pathogenic variant", r"deleterious variant", r"loss[- ]of[- ]function variant",
    r"mutation in", r"traced to a mutation", r"underlying cause .* (mutation|variant)",
    r"sequencing .* revealed .* (pathogenic|causal) variant",
    r"confirmed .* (mutation|variant)", r"yielded .* (mutation|variant)",
]
ASSOC_TRIGGERS = [
    r"associated with", r"linked to", r"related to", r"correlated with",
    r"presented with", r"features of", r"characterized by",
    r"clinical (features|profile) (include|includes)",
    r"the primary symptoms (were|was)", r"suffered from", r"complained of",
    r"with (a history of|evidence of)"
]
# 仅检测/提及（强排除）
MENTION_ONLY = [
    r"analysis of .* gene (was|were) performed",
    r"sequencing of .* gene (was|were) performed",
    r"genetic (analysis|testing) of .* gene",
    r"we sequenced .* gene", r"genetic screening of .* gene",
]

random.seed(SEED)

# ========= 1) Utilities =========
def sent_tokenize(text: str) -> List[Tuple[int,int,str]]:
    """简单分句：按 . ? ! ; 和换行切，返回 (start,end,sent_text) 列表"""
    spans, start = [], 0
    for m in re.finditer(r"[\.?!;]\s+|\n+", text):
        end = m.start()
        if end > start:
            spans.append((start, end, text[start:end]))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    return spans

def find_sentence_containing(span_a: Tuple[int,int], span_b: Tuple[int,int], sents):
    a_s, a_e = span_a; b_s, b_e = span_b
    for (s, e, sent) in sents:
        if (s <= a_s < e and s < a_e <= e) and (s <= b_s < e and s < b_e <= e):
            return (s, e, sent)
    return None

def find_sent_index_covering(span: Tuple[int,int], sents) -> int:
    s_idx, e_idx = span
    for i,(s,e,_) in enumerate(sents):
        if s <= s_idx < e and s < e_idx <= e:
            return i
    return -1

def get_window_text(sents, idx_a: int, idx_b: int, window: int = 1) -> Tuple[int,int,str]:
    """返回 (seg_start, seg_end, seg_text)"""
    lo = max(0, min(idx_a, idx_b) - window)
    hi = min(len(sents)-1, max(idx_a, idx_b) + window)
    seg_start = sents[lo][0]; seg_end = sents[hi][1]
    return seg_start, seg_end, " ".join(sents[i][2] for i in range(lo, hi+1))

def insert_markers(sentence: str, sent_start: int,
                   e1: Tuple[int,int,str], e2: Tuple[int,int,str]) -> str:
    (a_s, a_e, _), (b_s, b_e, _) = e1, e2
    a_s -= sent_start; a_e -= sent_start
    b_s -= sent_start; b_e -= sent_start
    pairs = sorted([(a_s,a_e,"E1"), (b_s,b_e,"E2")], key=lambda x: x[0])
    out, prev = [], 0
    for s, e, tag in pairs:
        out += [sentence[prev:s], f"[{tag}]", sentence[s:e], f"[/{tag}]"]; prev = e
    out.append(sentence[prev:])
    return "".join(out)

def regex_any(patterns: List[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def decide_relation_binary(ctx_text: str) -> str:
    """二分类判定：命中因果/关联且不是“仅提到/检测” → HasRelation"""
    t = ctx_text.lower()
    if regex_any(MENTION_ONLY, t):
        return "NoRelation"
    if regex_any(CAUSES_TRIGGERS, t) or regex_any(ASSOC_TRIGGERS, t):
        return "HasRelation"
    return "NoRelation"

@dataclass
class Span:
    start: int; end: int; label: str; text: str

def collect_spans(line_obj: Dict[str, Any]) -> List[Span]:
    spans, text = [], line_obj.get("text","")
    for sp in line_obj.get("spans", []):
        s = int(sp.get("start", -1)); e = int(sp.get("end", -1))
        lab = sp.get("label", "")
        if 0 <= s < e <= len(text):
            spans.append(Span(s,e,lab,text[s:e]))
    return spans

# ========= 2) Main conversion =========
def process_jsonl(file_path: str) -> List[Dict[str, Any]]:
    rows, fname = [], os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = obj.get("text","")
            spans = collect_spans(obj)
            if not spans or not text:
                continue

            sents = sent_tokenize(text)
            heads = [s for s in spans if s.label in HEAD_TYPES]
            tails = [s for s in spans if s.label in TAIL_TYPES]

            for h in heads:
                for t in tails:
                    # 1) 同句优先
                    same = find_sentence_containing((h.start,h.end),(t.start,t.end), sents)
                    if same is not None:
                        sent_start, sent_end, sent_text = same
                        rel = decide_relation_binary(sent_text)
                        text_marked = insert_markers(sent_text, sent_start,
                                                     (h.start,h.end,h.text),
                                                     (t.start,t.end,t.text))
                        rows.append({
                            "doc_id": f"{fname}::line{li}",
                            "sentence": sent_text.strip(),
                            "text_marked": text_marked.strip(),
                            "e1_text": h.text, "e1_type": h.label, "e1_start": h.start, "e1_end": h.end,
                            "e2_text": t.text, "e2_type": t.label, "e2_start": t.start, "e2_end": t.end,
                            "relation": rel,
                            "source": "gold"
                        })
                        continue

                    # 2) 相邻句窗口（±WINDOW_SENT）
                    hi = find_sent_index_covering((h.start,h.end), sents)
                    ti = find_sent_index_covering((t.start,t.end), sents)
                    if hi != -1 and ti != -1 and abs(hi - ti) <= WINDOW_SENT:
                        seg_s, seg_e, ctx_text = get_window_text(sents, hi, ti, window=0)
                        rel = decide_relation_binary(ctx_text)
                        # 用 head 所在句做可视化（简单安全）
                        vis_start, _, vis_text = sents[hi]
                        text_marked = insert_markers(vis_text, vis_start,
                                                     (h.start,h.end,h.text),
                                                     (t.start,t.end,t.text))
                        rows.append({
                            "doc_id": f"{fname}::line{li}",
                            "sentence": vis_text.strip(),
                            "text_marked": text_marked.strip(),
                            "e1_text": h.text, "e1_type": h.label, "e1_start": h.start, "e1_end": h.end,
                            "e2_text": t.text, "e2_type": t.label, "e2_start": t.start, "e2_end": t.end,
                            "relation": rel,
                            "source": "gold"
                        })
                        continue

                    # 3) 远距：跳过（避免噪声负例过多）
                    # 如需更激进的负例，可在此处构造 NoRelation，但建议保持为跳过。
    return rows

def build_dataset(input_dir: str) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    for fp in files:
        all_rows.extend(process_jsonl(fp))
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # 仅保留合法头尾类型
    df = df[(df["e1_type"].isin(HEAD_TYPES)) & (df["e2_type"].isin(TAIL_TYPES))].copy()

    # 负采样：按正例数量限制 NoRelation 数量
    pos_mask = df["relation"] == "HasRelation"
    no_mask  = df["relation"] == "NoRelation"
    n_pos, n_no = int(pos_mask.sum()), int(no_mask.sum())
    if n_pos > 0 and n_no > int(n_pos * NEG_RATIO):
        keep_no = df[no_mask].sample(n=int(n_pos * NEG_RATIO), random_state=SEED).index
        df = pd.concat([df[pos_mask], df.loc[keep_no]], axis=0).reset_index(drop=True)

    # 去重
    df = df.drop_duplicates(
        subset=["doc_id","e1_start","e1_end","e2_start","e2_end","relation","sentence"]
    ).reset_index(drop=True)
    return df

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = build_dataset(INPUT_DIR)
    if df.empty:
        print("No rows produced. Check INPUT_DIR and input JSONL format.")
        return

    cols = ["doc_id","sentence","text_marked",
            "e1_text","e1_type","e1_start","e1_end",
            "e2_text","e2_type","e2_start","e2_end",
            "relation","source"]
    df = df[cols]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CSV}  rows={len(df)}")
    print(df["relation"].value_counts())

if __name__ == "__main__":
    main()

# -------- check (抽查与分布) --------
csv_path = OUTPUT_CSV
if os.path.exists(csv_path):
    _df = pd.read_csv(csv_path)
    ok = _df[(_df["e1_type"].isin(["GENE","VARIANT","GENE_VARIANT"])) & (_df["e2_type"]=="HPO_TERM")]
    audit = ok.sample(n=min(20, len(ok)), random_state=42)
    audit.to_csv(os.path.join(os.path.dirname(csv_path), "audit_gold_20.csv"),
                 index=False, encoding="utf-8-sig")
    print(ok["relation"].value_counts())
    print(pd.crosstab(ok["e1_type"], ok["relation"]))
