import json, random

docs = [json.loads(l) for l in open("rag/library/library.jsonl", encoding="utf-8")]
fr = [d for d in docs if d["language"] == "fr"]     # 9538 passages
d = fr[random.randrange(len(fr))]                    # recommence tant que tu veux
print(d["id"], d["source"], "\n", d["text"])

hits = [d["id"] for d in fr if "Gao" in d["text"]]
print(hits)