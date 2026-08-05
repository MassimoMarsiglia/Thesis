from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

MODEL_NAME = "artifacts/models/embeddinggemma-document-linker/checkpoint-39"

EVAL_DIR = Path("data/enriched")


device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Load base model (no finetune)
# ----------------------------

model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
).to(device)

model.max_seq_length = 512


# ----------------------------
# Load IR evaluation data
# ----------------------------

queries_ds = load_dataset(
    "json",
    data_files=str(EVAL_DIR / "queries.jsonl"),
)["train"]

queries = {row["id"]: row["value"] for row in queries_ds}


corpus_ds = load_dataset(
    "json",
    data_files=str(EVAL_DIR / "corpus.jsonl"),
)["train"]

corpus = {row["id"]: row["value"] for row in corpus_ds}


relevant_ds = load_dataset(
    "json",
    data_files=str(EVAL_DIR / "relevant_docs.jsonl"),
)["train"]

relevant_docs = {row["id"]: set(row["value"]) for row in relevant_ds}


# ----------------------------
# Evaluator
# ----------------------------

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="embeddinggemma-augmented-zero-shot",
    accuracy_at_k=[1, 5, 10],
    mrr_at_k=[1, 5, 10],
    ndcg_at_k=[1, 5, 10],
    show_progress_bar=True,
)


# ----------------------------
# Run baseline
# ----------------------------

results = evaluator(model)

print(results)
