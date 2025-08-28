# -*- coding: utf-8 -*-
# predict_bin.py —— 从 jsonl 推理，输出预测 CSV（NoRelation/HasRelation）
import re, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "re_best_model_bin_mix"   # 换成你的模型目录
INPUT_JSONL = BASE / "silver.jsonl"      # 待推理 jsonl
OUT_CSV     = BASE / "pred_pairs.csv"

MAX_LEN = 256
BATCH = 64
THR_POS = 0.55        # 若你做过阈值扫描，用最佳阈值替换
NR_BIAS_FALLBACK = 0.80
HEADS = {"GENE","VARIANT","GENE_VARIANT"}; TAIL="HPO_TERM"

def sents(t):
    out=[]; st=0
    for m in re.finditer(r"[\.?!]\s+|\n+", t):
        ed=m.start(); 
        if ed>st: out.append((st,ed,t[st:ed])); 
        st=m.end()
    if st<len(t): out.append((st,len(t),t[st:]))
    return out

def cover(span, ss):
    s,e=span
    for i,(a,b,_) in enumerate(ss):
        if a<=s and e<=b: return i
    return -1

def window(t, ss, ai, bi, W=4):
    lo=max(0,min(ai,bi)-W); hi=min(len(ss)-1,max(ai,bi)+W)
    return ss[lo][0], ss[hi][1], t[ss[lo][0]:ss[hi][1]]

def mark(seg, seg_s, h, p):
    hs,he,ht=h; ts,te,tt=p
    hs,he=hs-seg_s,he-seg_s; ts,te=ts-seg_s,te-seg_s
    if hs>ts: (hs,he,ht),(ts,te,tt)=(ts,te,tt),(hs,he,ht)
    return seg[:hs]+"[E1]"+seg[hs:he]+"[/E1]"+seg[he:ts]+"[E2]"+seg[ts:te]+"[/E2]"+seg[te:]

def collect(jsonl):
    rows=[]
    for li,line in enumerate(open(jsonl,"r",encoding="utf-8")):
        if not line.strip(): continue
        try: obj=json.loads(line)
        except: continue
        text=obj.get("text",""); raw=obj.get("spans",[])
        if not text or not raw: continue
        ents=[]
        for sp in raw:
            try: s=int(sp["start"]); e=int(sp["end"]); lab=sp["label"]
            except: continue
            if 0<=s<e<=len(text): ents.append((s,e,lab,text[s:e]))
        ss=sents(text)
        heads=[x for x in ents if x[2] in HEADS]
        tails=[x for x in ents if x[2]==TAIL]
        for h in heads:
            ai=cover((h[0],h[1]),ss); 
            if ai==-1: continue
            for t in tails:
                bi=cover((t[0],t[1]),ss); 
                if bi==-1: continue
                if ai==bi: seg_s,seg_e,seg_txt=ss[ai][0],ss[ai][1],ss[ai][2]
                elif abs(ai-bi)<=4: seg_s,seg_e,seg_txt=window(text,ss,ai,bi,4)
                else: continue
                rows.append({
                    "doc_id":f"line{li}",
                    "sentence":seg_txt.strip(),
                    "text_marked":mark(seg_txt,seg_s,(h[0],h[1],h[3]),(t[0],t[1],t[3])).strip(),
                    "e1_text":h[3],"e1_type":h[2],"e1_start":h[0],"e1_end":h[1],
                    "e2_text":t[3],"e2_type":t[2],"e2_start":t[0],"e2_end":t[1],
                })
    return pd.DataFrame(rows)

def main():
    assert MODEL_DIR.exists(), f"模型不存在: {MODEL_DIR}"
    df=collect(INPUT_JSONL)
    if df.empty: 
        print("没有抽取到候选对"); return
    tok=AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    tok.add_special_tokens({"additional_special_tokens":["[E1]","[/E1]","[E2]","[/E2]"]})
    mdl=AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval().to("cpu")
    # NoRelation 偏置
    bias=NR_BIAS_FALLBACK
    bp=MODEL_DIR/"nr_bias.json"
    if bp.exists():
        try:
            import json
            bias=float(json.load(open(bp,"r",encoding="utf-8"))["no_relation_logit_bias"])
        except: pass

    xs=df["text_marked"].tolist(); all_logits=[]
    with torch.no_grad():
        for i in range(0,len(xs),BATCH):
            enc=tok(xs[i:i+BATCH], truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            lg=mdl(**enc).logits.detach().cpu().numpy()
            lg[:,0]-=bias
            all_logits.append(lg)
    logits=np.concatenate(all_logits,0)
    probs=torch.softmax(torch.tensor(logits),dim=-1).numpy()
    p_nr=probs[:,0]; p_pos=probs[:,1]
    pred=(p_pos>=THR_POS).astype(int)
    out=df.copy()
    out["p_NoRel"]=p_nr; out["p_HasRel"]=p_pos; out["pred"]=np.where(pred==1,"HasRelation","NoRelation")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved:", OUT_CSV, "rows=", len(out))
    print(out["pred"].value_counts())
if __name__=="__main__":
    main()
