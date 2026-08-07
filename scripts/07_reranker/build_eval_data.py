from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# -------------------------
# Paths
# -------------------------

CORPUS_PATH = "data/test/corpus.jsonl"
QUERY_PATH = "data/test/queries.jsonl"
RELEVANT_DOCS_PATH = "data/test/relevant_docs.jsonl"

MODEL_NAME = "artifacts/models/embeddinggemma-document-linker/checkpoint-170"

OUTPUT_PATH = "data/reranker/test/reranker_eval.jsonl"

TOP_K = 100


Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------
# Load datasets
# -------------------------

corpus = load_dataset(
    "json",
    data_files=CORPUS_PATH,
    split="train",
)

# {
#   "id": document_id,
#   "value": document text
# }


queries = load_dataset(
    "json",
    data_files=QUERY_PATH,
    split="train",
)

# {
#   "id": query_id,
#   "value": query text
# }


relevant_docs = load_dataset(
    "json",
    data_files=RELEVANT_DOCS_PATH,
    split="train",
)

# {
#   "id": query_id,
#   "value": [
#       relevant document ids
#   ]
# }


# -------------------------
# Build relevance lookup
# -------------------------

relevant_map = {row["id"]: set(row["value"]) for row in relevant_docs}


# -------------------------
# Load your embedding model
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
).to(device)


# -------------------------
# Encode corpus
# -------------------------

print("Encoding corpus...")


corpus_embeddings = model.encode(
    corpus["value"],
    normalize_embeddings=True,
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=True,
)


print(
    "Corpus embeddings:",
    corpus_embeddings.shape,
)


# -------------------------
# Encode queries
# -------------------------

print("Encoding queries...")


query_embeddings = model.encode(
    queries["value"],
    normalize_embeddings=True,
    convert_to_tensor=True,
    batch_size=64,
    show_progress_bar=True,
)


# -------------------------
# Retrieve candidates
# -------------------------

print("Running semantic search...")


results = semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=TOP_K,
)


# -------------------------
# Build CrossEncoder eval data
# -------------------------

eval_samples = []

missing_positive = 0


for query_row, hits in zip(
    queries,
    results,
):
    query_id = query_row["id"]
    query_text = query_row["value"]

    relevant_ids = relevant_map.get(
        query_id,
        set(),
    )

    documents = []
    positives = []

    for hit in hits:
        doc = corpus[hit["corpus_id"]]

        documents.append(doc["value"])

        if doc["id"] in relevant_ids:
            positives.append(doc["value"])

    # Reranker cannot be evaluated
    # if retrieval missed all positives
    if not positives:
        missing_positive += 1
        continue

    eval_samples.append(
        {
            "query": query_text,
            "positive": positives,
            "documents": documents,
        }
    )


# -------------------------
# Save
# -------------------------

dataset = Dataset.from_list(eval_samples)


dataset.to_json(
    OUTPUT_PATH,
    lines=True,
)


print(f"Saved {len(dataset)} evaluation queries")

print(f"Queries with no retrieved positive: {missing_positive}")


# print(dataset[0])
