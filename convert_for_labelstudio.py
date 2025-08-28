import json, argparse
from pathlib import Path
ALLOWED = {"associated_with","causes","not_associated","None"}

def main(in_jsonl: Path, out_json: Path):
    tasks=[]
    for line in in_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        ex=json.loads(line); text=ex["text"]; spans=ex["spans"]; rels=ex.get("relations",[])
        results=[]; idmap={}
        for i,sp in enumerate(spans):
            rid=f"r{i}"; idmap[i]=rid
            results.append({"id":rid,"from_name":"ner","to_name":"text","type":"labels",
                "value":{"start":sp["start"],"end":sp["end"],"text":text[sp["start"]:sp["end"]],"labels":[sp["label"]]}})
        for r in rels:
            lbl=r["label"]; 
            if lbl not in ALLOWED: lbl="None"
            results.append({"type":"relation","from_id":idmap[r["head"]],"to_id":idmap[r["child"]],
                            "labels":[lbl],"direction":"right"})
        tasks.append({"data":{"text":text},"predictions":[{"result":results}]})
    out_json.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", out_json, "tasks:", len(tasks))

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    a=ap.parse_args()
    main(Path(a.in_jsonl), Path(a.out_json))
