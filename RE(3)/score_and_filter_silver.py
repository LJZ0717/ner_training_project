# -*- coding: utf-8 -*-
"""
score_and_filter_silver_bin.py —— 二分类 (NoRelation / HasRelation)
- 读取 ./silver.jsonl
- 构造候选句（同句 + 相邻句窗口）
- 用 ./re_best_model_bin/ 打分
- 导出：
  - re_pairs_silver_scored_bin.csv(全量打分)
  - re_pairs_silver_kept_bin.csv(筛选后的银样本，仅 HasRelation,带 weight)
"""

import re, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ========= 相对路径 =========
BASE = Path(__file__).resolve().parent
INPUT_JSONL = BASE / "silver.jsonl"
MODEL_DIR   = BASE / "re_best_model_bin"
OUT_SCORED  = BASE / "re_pairs_silver_scored_bin.csv"
OUT_KEPT    = BASE / "re_pairs_silver_kept_bin.csv"

# ========= 超参（可按需要调整）=========
MAX_LEN      = 256
BATCH_SIZE   = 64
WINDOW_SENT  = 4     # 同句 + 相邻句窗口（±W）
# 初始阈值与边际（pos_prob: HasRelation 概率；margin: pos_prob - p_NoRel）
THR_POS      = 0.55
THR_MARGIN   = 0.00
RELAX_SCHEDULE = [
    (0.55, 0.00),
    (0.52, -0.02),
    (0.50, -0.05),
]
# 兜底：如果还筛不到，则从非 NoRelation 的样本里取 Top-K（按 margin 排序）
TOPK_FALLBACK_FRAC = 0.05  # 前 5%
TOPK_MIN = 80              # 至少 80 条（不够就有多少取多少）

# 若模型目录有 nr_bias.json，就对 NoRelation logit 做惩罚（更容易保留正类）
NR_BIAS_FALLBACK = 0.80

# 实体类型（与你的转换脚本一致）
HEAD_TYPES = {"GENE", "VARIANT", "GENE_VARIANT"}
TAIL_TYPE  = "HPO_TERM"

# ========= 小工具 =========
def sent_tokenize(text: str) -> List[Tuple[int,int,str]]:
    out=[]; start=0
    for m in re.finditer(r"[\.?!]\s+|\n+", text):
        end=m.start()
        if end>start: out.append((start,end,text[start:end]))
        start=m.end()
    if start<len(text): out.append((start,len(text),text[start:]))
    return out

def cover_idx(span, sents):
    s,e = span
    for i,(ss,ee,_) in enumerate(sents):
        if ss<=s and e<=ee: return i
    return -1

def window_text(text, sents, ai, bi, W):
    lo=max(0,min(ai,bi)-W); hi=min(len(sents)-1,max(ai,bi)+W)
    return sents[lo][0], sents[hi][1], text[sents[lo][0]:sents[hi][1]]

def insert_mark(seg_text, seg_start, h, t):
    hs,he,ht = h; ts,te,tt = t
    hs,he=hs-seg_start, he-seg_start
    ts,te=ts-seg_start, te-seg_start
    if hs>ts: (hs,he,ht),(ts,te,tt)=(ts,te,tt),(hs,he,ht)
    return "".join([seg_text[:hs],"[E1]",seg_text[hs:he],"[/E1]",
                    seg_text[he:ts],"[E2]",seg_text[ts:te],"[/E2]",seg_text[te:]])

def collect_pairs(jsonl_path: Path, W: int) -> pd.DataFrame:
    rows=[]
    with open(jsonl_path,"r",encoding="utf-8") as f:
        for li,line in enumerate(f):
            line=line.strip()
            if not line: continue
            try: obj=json.loads(line)
            except: continue
            text=obj.get("text",""); raw=obj.get("spans",[])
            if not text or not raw: continue

            spans=[]
            for sp in raw:
                try:
                    s=int(sp.get("start",-1)); e=int(sp.get("end",-1)); lab=sp.get("label","")
                except: continue
                if 0<=s<e<=len(text):
                    spans.append((s,e,lab,text[s:e]))

            sents=sent_tokenize(text)
            heads=[x for x in spans if x[2] in HEAD_TYPES]
            tails=[x for x in spans if x[2]==TAIL_TYPE]
            if not heads or not tails: continue

            for hs,he,hlab,htxt in heads:
                ai=cover_idx((hs,he),sents)
                if ai==-1: continue
                for ts,te,tlab,ttxt in tails:
                    bi=cover_idx((ts,te),sents)
                    if bi==-1: continue

                    if ai==bi:
                        seg_s,seg_e,seg_txt=sents[ai][0],sents[ai][1],sents[ai][2]
                    elif abs(ai-bi)<=W:
                        seg_s,seg_e,seg_txt=window_text(text,sents,ai,bi,W)
                    else:
                        continue

                    rows.append({
                        "doc_id":f"line{li}",
                        "sentence":seg_txt.strip(),
                        "text_marked":insert_mark(seg_txt,seg_s,(hs,he,htxt),(ts,te,ttxt)).strip(),
                        "e1_text":htxt,"e1_type":hlab,"e1_start":hs,"e1_end":he,
                        "e2_text":ttxt,"e2_type":tlab,"e2_start":ts,"e2_end":te
                    })

    df=pd.DataFrame(rows)
    if not df.empty:
        df.drop_duplicates(subset=["doc_id","e1_start","e1_end","e2_start","e2_end","sentence"],
                           inplace=True, ignore_index=True)
    return df

# ========= 主流程 =========
def main():
    print("工作目录：", BASE)
    if not INPUT_JSONL.exists(): raise FileNotFoundError(f"找不到 silver.jsonl：{INPUT_JSONL}")
    if not MODEL_DIR.exists():   raise FileNotFoundError(f"找不到模型目录：{MODEL_DIR}")

    # 1) 收集候选
    df = collect_pairs(INPUT_JSONL, WINDOW_SENT)
    if df.empty:
        print("未构造出候选对；可把 WINDOW_SENT 调大（如 5）。"); return
    print("候选数：", len(df))

    # 2) 加载模型
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    tok.add_special_tokens({"additional_special_tokens":["[E1]","[/E1]","[E2]","[/E2]"]})
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval().to("cpu")

    # 读取 nr_bias.json（如无则用兜底）
    nr_bias = NR_BIAS_FALLBACK
    bp = MODEL_DIR / "nr_bias.json"
    if bp.exists():
        try:
            nr_bias = float(json.load(open(bp,"r",encoding="utf-8")).get("no_relation_logit_bias", nr_bias))
            print(f"应用 NoRelation 偏置：{nr_bias:.2f}")
        except:
            print(f"读取 nr_bias.json 失败，使用兜底偏置：{nr_bias:.2f}")
    else:
        print(f"未找到 nr_bias.json，使用兜底偏置：{nr_bias:.2f}")

    # 3) 批量推理
    texts=df["text_marked"].astype(str).tolist()
    logits_all=[]
    with torch.no_grad():
        for i in range(0,len(texts),BATCH_SIZE):
            enc=tok(texts[i:i+BATCH_SIZE], truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt")
            out=mdl(**enc)
            lg=out.logits.detach().cpu().numpy()
            # 二分类：[0]=NoRelation, [1]=HasRelation
            lg[:,0] -= nr_bias  # 压低 NoRelation
            logits_all.append(lg)
    logits=np.concatenate(logits_all, axis=0)
    probs=torch.softmax(torch.tensor(logits), dim=-1).numpy()
    p_nr = probs[:,0]
    p_pos= probs[:,1]
    margin = p_pos - p_nr

    # 4) 导出全量打分
    df_s = df.copy()
    df_s["p_NoRel"]  = p_nr
    df_s["p_HasRel"] = p_pos
    df_s["margin"]   = margin
    df_s["pred"]     = (p_pos >= 0.5).astype(int)
    df_s.to_csv(OUT_SCORED, index=False, encoding="utf-8-sig")
    print(f"已保存全量打分：{OUT_SCORED} rows={len(df_s)}")
    print(df_s["pred"].value_counts().rename({0:"NoRel",1:"HasRel"}))

    # 5) 按阈值筛选 + 自适应降阈
    kept=None
    for pos_th, mar_th in RELAX_SCHEDULE:
        mask = (df_s["p_HasRel"]>=pos_th) & (df_s["margin"]>=mar_th)
        cand = df_s[mask].copy()
        if len(cand)>0:
            print(f"采用阈值 pos>={pos_th:.2f}, margin>={mar_th:.2f} → kept={len(cand)}")
            kept = cand; break
        else:
            print(f"阈值 pos>={pos_th:.2f}, margin>={mar_th:.2f} 下为 0，继续放宽…")

    # 兜底：从非 NoRel 的样本中按 margin 取 Top-K
    if kept is None or len(kept)==0:
        pos = df_s[df_s["p_HasRel"]>df_s["p_NoRel"]].copy()
        if len(pos)>0:
            K = max(int(len(pos)*TOPK_FALLBACK_FRAC), TOPK_MIN)
            kept = pos.sort_values(["margin","p_HasRel"], ascending=False).head(K).copy()
            print(f"Top-K 兜底：取前 {len(kept)} 条正类候选。")
        else:
            kept = df_s.iloc[0:0].copy()
            print("仍为 0 条；建议调大 NR_BIAS_FALLBACK 或 WINDOW_SENT。")

    # 6) 导出训练所需列（全标为 HasRelation）
    kept["relation"]  = "HasRelation"
    kept["is_silver"] = 1
    kept["weight"]    = 0.5
    cols = ["doc_id","sentence","text_marked",
            "e1_text","e1_type","e1_start","e1_end",
            "e2_text","e2_type","e2_start","e2_end",
            "relation","p_HasRel","p_NoRel","margin",
            "is_silver","weight"]
    kept = kept[cols].drop_duplicates(ignore_index=True)
    kept.to_csv(OUT_KEPT, index=False, encoding="utf-8-sig")
    print(f"已保存筛选结果：{OUT_KEPT} rows={len(kept)}")
    if len(kept)>0:
        print(kept["relation"].value_counts())

if __name__=="__main__":
    main()
