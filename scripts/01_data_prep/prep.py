import json
import re

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

REFS_FILE = "data/refs.jsonl"
DOCS_FILE = "data/docs.jsonl"
OUTPUT = "artifacts/bm25_candidates.jsonl"

TOP_K = 20


def tokenize(text):
    if not isinstance(text, str):
        return []

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# -----------------------
# Load data
# -----------------------

refs = pd.read_json(REFS_FILE, lines=True)

docs = pd.read_json(DOCS_FILE, lines=True)


print("Refs:")
print(refs.head())

print("Docs:")
print(docs.head())


# -----------------------
# Build BM25 index
# -----------------------

refs = refs[refs["description"].notna()].reset_index(drop=True)

ref_ids = refs["source_build_id"].tolist()
ref_descriptions = refs["description"].tolist()

corpus = [tokenize(x) for x in ref_descriptions]

print(f"Indexing {len(corpus)} refs")

bm25 = BM25Okapi(corpus)


# -----------------------
# Search docs and stream output
# -----------------------

docs = docs[docs["short_title"].notna()]

count = 0

with open(OUTPUT, "w") as f:
    for _, doc in tqdm(docs.iterrows(), total=len(docs)):
        query = tokenize(doc["short_title"])

        if not query:
            continue

        scores = bm25.get_scores(query)

        top_indices = scores.argsort()[-TOP_K:][::-1]

        for idx in top_indices:
            score = float(scores[idx])

            if score <= 0:
                continue

            result = {
                "build_id": doc["build_id"],
                "ref_id": ref_ids[idx],
                "short_title": doc["short_title"],
                "description": ref_descriptions[idx],
                "bm25_score": score,
            }

            f.write(json.dumps(result) + "\n")

            count += 1


print(f"Saved {count} candidates to {OUTPUT}")
