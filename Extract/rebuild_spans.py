
import pandas as pd

REPAIRED_CLEAN = r"C:\Users\Administrator\Desktop\Project\Extract\pred_pairs_entities_repaired_clean.csv"
OUT_SPAN       = r"C:\Users\Administrator\Desktop\Project\Extract\extracted_spans_from_repaired.csv"

df = pd.read_csv(REPAIRED_CLEAN, encoding="utf-8")
# 如果有 pred 列就只取正例；没有就全取
df = df[df["pred"]=="HasRelation"] if "pred" in df.columns else df

spans = df[["doc_id","e2_text","sentence"]].rename(
    columns={"e2_text":"span_text","sentence":"context"}
)
spans = spans[spans["span_text"].astype(str).str.strip().str.len()>0].drop_duplicates()

spans.to_csv(OUT_SPAN, index=False, encoding="utf-8-sig", quoting=1)  # 全加引号，防止换行/逗号破坏
print("saved ->", OUT_SPAN, "rows:", len(spans))
