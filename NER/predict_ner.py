import os, sys, re, numbers
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF


BASE_DIR   = r"C:\Users\Administrator\Desktop\Project\NER"
MODEL_PATH = os.path.join(BASE_DIR, "best_BioBERT_model.pt")
LABEL_MAP  = os.path.join(BASE_DIR, "label_mapping.pt")
INPUT_TXT  = os.path.join(BASE_DIR, "test.txt")
OUTPUT_CSV = os.path.join(BASE_DIR, "ner_predictions.csv")

# -------- Fixed use of BioBERT --------
BIOBERT_NAME = "dmis-lab/biobert-base-cased-v1.1"
BATCH_SIZE   = 16
MAX_LENGTH   = 256           
SPLIT_LONG   = True          

# ---------------- Model  ----------------
class BertCRF(nn.Module):
    def __init__(self, model_name: str, num_tags: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def emissions(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        logits = self.classifier(out)  # (B, L, num_tags)
        return logits

    def decode(self, input_ids, attention_mask):
        emissions = self.emissions(input_ids, attention_mask)
        return self.crf.decode(emissions, mask=attention_mask.bool())  # List[List[int]]

# ---------------- Parsing label_mapping.pt ----------------
def _extract_int(x):
    if isinstance(x, numbers.Number):
        return int(x)
    if isinstance(x, str):
        return int(x) if x.isdigit() else int(float(x))
    if isinstance(x, dict):
        for key in ("id", "index", "label_id", "tag_id", "value"):
            if key in x:
                return _extract_int(x[key])
        if len(x) == 1:
            return _extract_int(next(iter(x.values())))
    raise ValueError(f"Cannot extract int from: {x}")

def parse_label_map(lm_obj):
    
    if isinstance(lm_obj, dict):
        for key in ("label2id", "label2idx", "labels_to_ids"):
            if key in lm_obj and isinstance(lm_obj[key], dict):
                l2i = {str(k): _extract_int(v) for k, v in lm_obj[key].items()}
                return {int(v): str(k) for k, v in l2i.items()}
        
        for key in ("id2label", "idx2label", "ids_to_labels"):
            if key in lm_obj:
                raw = lm_obj[key]
                if isinstance(raw, dict):
                    return { _extract_int(k): str(v) for k, v in raw.items() }
                if isinstance(raw, (list, tuple)):
                    return { i: str(v) for i, v in enumerate(raw) }
        
        if all(isinstance(k, str) for k in lm_obj.keys()):
            l2i = {str(k): _extract_int(v) for k, v in lm_obj.items()}
            return {int(v): str(k) for k, v in l2i.items()}
        try:
            return { _extract_int(k): str(v) for k, v in lm_obj.items() }
        except Exception:
            pass
    if isinstance(lm_obj, (list, tuple)):
        return { i: str(v) for i, v in enumerate(lm_obj) }
    raise ValueError("Unrecognized label map structure")

# ---------------- sentence division to avoid truncation of long paragraphs ----------------
_SENT_SPLIT = re.compile(r'(?<=[\.\!\?;；。！？”])\s+')
def split_long_line(line, max_len=MAX_LENGTH//2):
    if not SPLIT_LONG:
        return [line]
    # First break sentences by punctuation, then cut by length
    parts = [p.strip() for p in _SENT_SPLIT.split(line) if p.strip()]
    out = []
    for p in parts or [line]:
        if len(p) <= max_len:
            out.append(p)
        else:
            
            for i in range(0, len(p), max_len):
                out.append(p[i:i+max_len])
    return out

# ---------------- BIO -> spans ----------------
def bio_to_spans(tags, offsets, text):
    spans, i, n = [], 0, len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag.split("-", 1)[1]
            start = offsets[i][0]
            end   = offsets[i][1]
            j = i + 1
            while j < n and tags[j].startswith("I-"):
                end = offsets[j][1]
                j += 1
            if end > start:
                spans.append({
                    "span": text[start:end],
                    "label": label,
                    "start_char": int(start),
                    "end_char": int(end),
                    "context": text
                })
            i = j
        else:
            i += 1
    return spans

def main():
    print("=== NER Inference (BioBERT+CRF, hard-coded) ===")
    print("Python:", sys.executable)
    print("Model:", MODEL_PATH)
    print("Label map:", LABEL_MAP)
    print("Input:", INPUT_TXT)
    print("Output:", OUTPUT_CSV)

    
    for p in [MODEL_PATH, LABEL_MAP, INPUT_TXT]:
        if not os.path.isfile(p):
            print(f"[ERROR] File not found: {p}")
            pd.DataFrame(columns=["doc_id","span","label","start_char","end_char","context"]).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Read label mapping
    lm = torch.load(LABEL_MAP, map_location="cpu")
    id2label = parse_label_map(lm)
    id2label = {int(k): str(v) for k, v in id2label.items()}
    num_tags = max(id2label.keys()) + 1
    print("Num tags:", num_tags, "Sample:", dict(list(id2label.items())[:5]))

    # Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_NAME, use_fast=True)
    model = BertCRF(BIOBERT_NAME, num_tags=num_tags)
    state = torch.load(MODEL_PATH, map_location="cpu")
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    print("Model loaded.")

    # Read txt, each line can be divided into sentences
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    texts = []
    parents = []  # Record original line number
    for idx, line in enumerate(raw_lines):
        chunks = split_long_line(line)
        texts.extend(chunks)
        parents.extend([idx]*len(chunks))
    print(f"Loaded lines: {len(raw_lines)} -> chunks: {len(texts)}")

    results = []
    bs = BATCH_SIZE
    for i in range(0, len(texts), bs):
        batch_texts = texts[i:i+bs]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]

        with torch.no_grad():
            pred_ids_batch = model.decode(input_ids, attention_mask)

        for b_idx, pred_ids in enumerate(pred_ids_batch):
            seq_offsets = offsets[b_idx].tolist()
            kept_offsets = [off for off in seq_offsets if not (off[0] == 0 and off[1] == 0)]
            # Alignment length
            if len(pred_ids) > len(kept_offsets):
                pred_ids = pred_ids[:len(kept_offsets)]
            elif len(pred_ids) < len(kept_offsets):
                kept_offsets = kept_offsets[:len(pred_ids)]
            tags = [id2label.get(int(tid), "O") for tid in pred_ids]
            sent_text = batch_texts[b_idx]
            spans = bio_to_spans(tags, kept_offsets, sent_text)
            for sp in spans:
                sp["doc_id"] = int(parents[i + b_idx])  
                results.append(sp)

    out_df = pd.DataFrame(results, columns=["doc_id","span","label","start_char","end_char","context"])
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("Extraction done. Total spans:", len(out_df))
    print("Saved to:", OUTPUT_CSV)
    if len(out_df) > 0:
        print(out_df.head(5))

if __name__ == "__main__":
    main()
