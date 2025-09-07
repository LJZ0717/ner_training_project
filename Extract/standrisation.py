import os
import json
import pandas as pd

BASE_DIR         = r"C:\Users\Administrator\Desktop\Project\Extract"
IN_MAPPED        = os.path.join(BASE_DIR, "mapped_hpo.csv")

OUT_CLEAN        = os.path.join(BASE_DIR, "mapped_hpo_clean.csv")
OUT_AUTO         = os.path.join(BASE_DIR, "mapped_hpo_auto_accept.csv")
OUT_REVIEW       = os.path.join(BASE_DIR, "mapped_hpo_review.csv")
OUT_RETRY        = os.path.join(BASE_DIR, "mapped_hpo_to_retry.csv")
OUT_CONFLICTS    = os.path.join(BASE_DIR, "mapped_hpo_conflicts.csv")
OUT_SUMMARY_JSON = os.path.join(BASE_DIR, "mapped_hpo_summary.json")

# 阈值策略
THRESHOLD_CLEAR  = 88.0   # # Below this threshold => considered unmapped (clear ID/method, score=0)
REVIEW_MIN       = 90.0   # Lower limit of review interval (inclusive)
AUTO_ACCEPT_MIN  = 95.0   # Automatically pass the lower limit (inclusive)
# =========================================

def ensure_cols(df):
    base_cols = ["doc_id","span_text","best_hpo_id","best_hpo_name","score","method","candidates_json"]
    for c in base_cols:
        if c not in df.columns:
            if c == "score":
                df[c] = 0.0
            else:
                df[c] = ""
    return df

def main():
    assert os.path.exists(IN_MAPPED), f"Not found: {IN_MAPPED}"
    df = pd.read_csv(IN_MAPPED, encoding="utf-8")
    df = ensure_cols(df)

    # ---- 1) 
    method_empty = (df["method"].astype(str).str.len() == 0) | (df["method"].isna())
    low_conf     = (df["score"].fillna(0) < THRESHOLD_CLEAR)
    mask_clear   = method_empty | low_conf

    df_clean = df.copy()
    df_clean.loc[mask_clear, ["best_hpo_id","best_hpo_name","method"]] = ""
    df_clean.loc[mask_clear, "score"] = 0.0

    # Do not allow score=0 but still have ID
    bad_mask = (df_clean["score"] <= 0.0) & (df_clean["best_hpo_id"].astype(str).str.len() > 0)
    df_clean.loc[bad_mask, ["best_hpo_id","best_hpo_name","method"]] = ""

    # ---- 2) Bucket ----
    auto_accept = df_clean[df_clean["score"] >= AUTO_ACCEPT_MIN].copy()
    review      = df_clean[(df_clean["score"] >= REVIEW_MIN) & (df_clean["score"] < AUTO_ACCEPT_MIN)].copy()
    to_retry    = df_clean[df_clean["score"] < REVIEW_MIN].copy()

    # ---- 3) Export ----
    df_clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")
    auto_accept.to_csv(OUT_AUTO, index=False, encoding="utf-8-sig")
    review.to_csv(OUT_REVIEW, index=False, encoding="utf-8-sig")
    to_retry.to_csv(OUT_RETRY, index=False, encoding="utf-8-sig")

    # ---- 4) Conflict detection: same doc_id+span_text → multiple IDs ----
    grp = df_clean.groupby(["doc_id","span_text"])["best_hpo_id"].nunique(dropna=True).reset_index()
    conflicts_grp = grp[grp["best_hpo_id"] > 1].sort_values("best_hpo_id", ascending=False)
    if not conflicts_grp.empty:
        keys = set(map(tuple, conflicts_grp[["doc_id","span_text"]].values))
        conflict_rows = df_clean[df_clean.apply(lambda r: (r["doc_id"], r["span_text"]) in keys, axis=1)]
        conflict_rows.to_csv(OUT_CONFLICTS, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=df_clean.columns).to_csv(OUT_CONFLICTS, index=False, encoding="utf-8-sig")

    # 5) 
    total = len(df_clean)
    unmapped = int((df_clean["best_hpo_id"].astype(str).str.len()==0).sum())
    n_auto  = len(auto_accept)
    n_rev   = len(review)
    n_retry = len(to_retry)
    n_conf  = 0 if conflicts_grp.empty else int(conflicts_grp.shape[0])

    summary = {
        "total_rows": total,
        "unmapped_rows": unmapped,
        "auto_accept_rows": n_auto,
        "review_rows": n_rev,
        "to_retry_rows": n_retry,
        "conflict_groups": n_conf,
        "thresholds": {
            "clear_below": THRESHOLD_CLEAR,
            "review_min": REVIEW_MIN,
            "auto_accept_min": AUTO_ACCEPT_MIN
        },
        "outputs": {
            "clean": OUT_CLEAN,
            "auto_accept": OUT_AUTO,
            "review": OUT_REVIEW,
            "to_retry": OUT_RETRY,
            "conflicts": OUT_CONFLICTS
        }
    }
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== POSTPROCESS DONE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
