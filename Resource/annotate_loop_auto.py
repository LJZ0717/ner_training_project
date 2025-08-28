import os
import subprocess
import sys

# 📁 Define folders
jsonl_dir = "jsonl_docs"
annotations_dir = "annotations"
os.makedirs(annotations_dir, exist_ok=True)

# 📦 Choose the SpaCy model
try:
    import spacy
    spacy.load("en_core_web_trf")
    spacy_model = "en_core_web_trf"
except:
    print("⚠️ en_core_web_trf not available. Falling back to en_core_web_sm.")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    spacy_model = "en_core_web_sm"

# 📋 Get all input files
input_files = sorted(f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl"))

# 🔁 Loop through each document
for filename in input_files:
    doc_id = filename.replace("_doc.jsonl", "")
    dataset_name = f"polg_{doc_id}"
    output_path = os.path.join(annotations_dir, f"{dataset_name}_spans.jsonl")

    # ✅ Skip if already saved
    if os.path.exists(output_path):
        print(f"✅ Already annotated: {filename} — skipping.")
        continue

    print(f"🚀 Starting annotation for: {filename}")

# 📦 Choose the SpaCy model
    try:
        subprocess.run([
            "prodigy",
            "ner.manual",  # or "ner.manual"
            dataset_name,
            spacy_model,
            os.path.join(jsonl_dir, filename),
            "--label",
            "PATIENT,GENE,HPO_TERM,GENE_VARIANT,AGE_ONSET,AGE_DEATH,AGE_FOLLOWUP"
        ])
    except KeyboardInterrupt:
        print("🛑 Annotation manually stopped.")
        break

    # 💾 Export the annotation
    print(f"💾 Exporting dataset: {dataset_name}")
    subprocess.run([
        "prodigy",
        "db-out",
        dataset_name,
        "--output", output_path
    ])
    print(f"✅ Saved annotations to: {output_path}\n")
