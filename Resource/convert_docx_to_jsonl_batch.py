import os
import json
import re
import spacy
from docx import Document

# Load the same tokenizer as used in Prodigy
nlp = spacy.load("en_core_web_trf")

input_folder = "/Users/thilokamiller/Desktop/Paediatric ST6/Academic/EBI_POLG"
output_folder = "/Users/thilokamiller/prodigy-demo/jsonl_docs"
os.makedirs(output_folder, exist_ok=True)

MAX_CHARS = 600

def clean_text(text):
    text = text.replace("\n", " ").strip()
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # fix hyphenation
    text = re.sub(r'\s+', ' ', text)
    return text

def chunk_paragraphs(paragraphs, max_chars):
    chunks = []
    current = ""
    for para in paragraphs:
        para = clean_text(para)
        if len(current) + len(para) + 1 <= max_chars:
            current += " " + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks

for filename in os.listdir(input_folder):
    if filename.endswith(".docx"):
        path = os.path.join(input_folder, filename)
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(output_folder, f"{base}.jsonl")

        try:
            docx = Document(path)
            paras = [p.text for p in docx.paragraphs if p.text.strip()]
            chunks = chunk_paragraphs(paras, MAX_CHARS)

            with open(out_path, "w") as f_out:
                for chunk in chunks:
                    doc = nlp(chunk)
                    tokens = [
                        {
                            "text": t.text,
                            "start": t.idx,
                            "end": t.idx + len(t.text),
                            "id": i,
                            "ws": bool(t.whitespace_)
                        }
                        for i, t in enumerate(doc)
                    ]
                    f_out.write(json.dumps({
                        "text": chunk,
                        "tokens": tokens,
                        "meta": {"source": base}
                    }) + "\n")
            print(f"✅ {filename} → {out_path}")
        except Exception as e:
            print(f"⚠️ Skipped {filename}: {e}")
