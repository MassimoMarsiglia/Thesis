# Linker — Regulatory Entity Linking System

An end-to-end system for automatically linking references in EU regulatory documents to their corresponding canonical entities. This is the codebase for the Bachelor's thesis:

> **"Evaluation of Modern Hopfield Networks as Associative Memory for Compliance Processes: A Comparative Study with Alternative Retrieval Architectures"**
> Massimo Diego Marsiglia — Hochschule fuer Technik und Wirtschaft Berlin

## Overview

The system solves the problem of matching references (which appear in varying surface forms — shortened titles, acronyms, identifiers, etc.) within regulatory builds to their canonical source documents. It uses a two-stage retrieval architecture:

1. **Bi-encoder** (dense embedding) — retrieves candidate documents via FAISS
2. **Cross-encoder** (reranker) — scores query-document pairs for fine-grained relevance

## Project Structure

```
linker/
├── main.py                  # Entry point (placeholder)
├── pyproject.toml           # Project dependencies (uv)
├── data/                    # All datasets and artifacts
│   ├── positive_pairs.jsonl # Raw positive reference-entity pairs
│   ├── docs.jsonl / refs.jsonl # All regulatory docs & references
│   ├── *_pairs.jsonl        # Train / eval / test splits
│   ├── enriched_*           # LLM-augmented training data
│   ├── mined_*              # Pairs with hard negatives
│   ├── combined/            # Combined original + augmented data
│   ├── train/ / eval/ / test/ # IR-formatted datasets (queries, corpus, relevant_docs)
│   ├── reranker/            # Cross-encoder evaluation data
│   └── data-duckdb.duckdb   # Source DuckDB database
├── scripts/                 # End-to-end pipeline (stages 01–08)
│   ├── 01_data_prep/        # Data extraction & cleaning from DuckDB
│   ├── 02_split_data/       # Train/eval/test splitting by document group
│   ├── 03_visualize/        # (empty — planned data visualizations)
│   ├── 04_augment/          # LLM-based synthetic data augmentation
│   ├── 05_build_pairs/      # Pair formatting, hard negative mining
│   ├── 06_train/            # Bi-encoder fine-tuning
│   ├── 07_eval/             # Evaluation (baseline vs trained, seen vs unseen)
│   ├── 07_reranker/         # Cross-encoder training & evaluation
│   └── 08_inference/        # FastAPI inference server
├── artifacts/               # Saved models, indices, checkpoints
│   ├── models/              # Fine-tuned bi-encoder & cross-encoder
│   ├── archive/             # Training checkpoints (checkpoint-9, -39, -78, -170)
│   ├── index.faiss          # Saved FAISS search index
│   └── images/              # Data split visualizations
├── config/                  # Configuration (currently empty)
├── thesis/                  # Bachelor thesis LaTeX source
│   ├── main.tex             # Main thesis document
│   ├── chapters/            # Chapter source files
│   ├── pictures/            # Thesis images/diagrams
│   └── references/          # Bibliography & acronyms
└── bin/                     # (empty)
```

## Pipeline Stages

### 01 — Data Preparation (`scripts/01_data_prep/`)

Extracts positive training pairs from the raw EUR-Lex regulatory dataset stored in DuckDB.

- **`build_dataset.sql`** — Core SQL pipeline that cleans the dataset, extracts references and document metadata, mines rare-word matches, applies Jaro-Winkler similarity, extracts document identifiers, and classifies reference contexts (amending, repealing, implementing, etc.). Outputs `positive_pairs.jsonl`.
- **`prep.py`** — Builds a BM25 index over reference descriptions and retrieves top-20 candidates per document. Outputs `bm25_candidates.jsonl`.

### 02 — Data Splitting (`scripts/02_split_data/`)

Splits positive pairs into train (80%), validation (13%), and test (7%) sets by unique document groups to prevent data leakage.

### 03 — Visualization (`scripts/03_visualize/`)

Planned directory for data visualizations (currently empty).

### 04 — Augmentation (`scripts/04_augment/`)

Uses an LLM to generate synthetic training examples that mimic real-world noise patterns in regulatory references.

- **`extract_noise_patterns.py`** — Asks an LLM to build a taxonomy of title and description patterns (shortened, partial, acronym, boilerplate-removed, etc.). Outputs `noise_patterns.jsonl`.
- **`enrich.py`** — For each positive pair, generates 3–5 synthetic variants following the noise taxonomy. Uses async streaming with concurrency control. Outputs `enriched_training_pairs.jsonl`.

### 05 — Pair Building (`scripts/05_build_pairs/`)

Formats raw pairs into the structure expected by SBERT-style training and creates IR evaluation datasets.

- **`build_pairs.py`** — Transforms pairs into anchor/positive format, creates IR datasets (queries, corpus, relevant_docs) for eval and test splits.
- **`combine_pairs.py`** — Concatenates original and augmented training pairs into a unified dataset.
- **`expand_enriched_pairs.py`** — Expands LLM-generated enriched examples into individual training pairs.
- **`mine_negatives/`** — Uses a trained embedding model to mine hard negatives (semantically similar but incorrect documents) via FAISS. Outputs `mined_training_pairs_v2.jsonl`.

### 06 — Training (`scripts/06_train/`)

Fine-tunes the bi-encoder embedding model for dense retrieval.

- **`train.py`** — Fine-tunes `google/embeddinggemma-300m` using `CachedMultipleNegativesRankingLoss` with 8-bit AdamW optimizer. Trains for 2 epochs with batch size 128, gradient accumulation 4. Evaluates using InformationRetrievalEvaluator (Accuracy@K, MRR@K, NDCG@K for K=[1,5,10]).
- **`baseline.py`** — Evaluates the base unfine-tuned model as a zero-shot baseline.

### 07 — Reranker (`scripts/07_reranker/`)

Trains a cross-encoder for second-stage reranking.

- **`build_eval_data.py`** — Uses the trained bi-encoder to retrieve top-100 candidates per query, then builds reranker evaluation samples.
- **`llm.py`** — Trains a cross-encoder (`BAAI/bge-reranker-v2-m3`) on positive/negative pairs using BinaryCrossEntropyLoss. Evaluated with CERerankingEvaluator (NDCG@1).
- **`xgboost.py`** — Commented-out prototype for XGBoost-based reranking.

### 07_eval — Augmented Evaluation (`scripts/07_eval/augmented/`)

A 2×2 evaluation matrix: base vs. trained model × seen vs. unseen augmented data. All scripts use `InformationRetrievalEvaluator`.

### 08 — Inference (`scripts/08_inference/`)

FastAPI REST server for real-time entity linking.

- **`main.py`** — Exposes `POST /link` endpoint accepting a reference document (title, description, relationship_type) and returning ranked candidate matches.
- **`pipeline.py`** — Two-stage retrieval: format input → encode with bi-encoder → FAISS search (top-50) → cross-encoder rerank.
- **`index.py`** — FAISS index management (build from scratch or load saved index).
- **`eval.py`** — Standalone evaluation of the full pipeline (Recall@50, Accuracy@1, MRR@50, NDCG@50).
- **`models/biencoder.py`** — SentenceTransformer wrapper.
- **`models/crossencoder.py`** — CrossEncoder wrapper.
- **`schemas.py`** — Pydantic request/response schemas.

## Key Dependencies

| Category | Packages |
|---|---|
| Deep Learning | torch, transformers, peft, trl, unsloth, torchao |
| Embeddings | sentence-transformers, faiss-gpu |
| Retrieval | rank-bm25, datasets, evaluate |
| Serving | fastapi, uvicorn |
| Data | pandas, numpy, scikit-learn, duckdb |
| LLM | openai, huggingface-hub |

## Quick Start

```bash
# Install dependencies (requires Python >=3.12)
uv sync

# Run the inference server
uv run scripts/08_inference/main.py
```

## Thesis

The LaTeX source for the Bachelor's thesis is in `thesis/`. The compiled PDF (`main.pdf`) is included in the repo.

```bash
cd thesis
# Compile the thesis
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

# Dataset:
All data was provided to by the partner company Fairfield & Archer.
The provided consisted of roughly 8000 eurolex regulatory documents + extracted features.
Language EN
Specific features of the dataset were used to train a regulatory document entity linking framework.
In other words the data was used to train on

the full data set is roughly 4.8GB in size the initial dataset came in form of a jsonl file which I converted to a DuckDB file.

All data was scraped from the official Eurolex website throughout 2024-2026 and was refined using an LLM (to my knowledge mainly GPT 4o bur I do not have access to the specifics)

The used data has been manually vetted by Fairfield & Archer and clients.

The rights to all the used data lie with Fairfield & Archer.

Given that 