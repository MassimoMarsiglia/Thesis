from pathlib import Path

import bitsandbytes as bnb
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

# =====================================================
# Config
# =====================================================

MODEL_NAME = "artifacts/models/embeddinggemma-document-linker/checkpoint-44"

OUTPUT_DIR = "artifacts/models/embeddinggemma-document-linker"
TRAIN_DIR = "data/combined/mined_training_pairs_v2.jsonl"
EVAL_DIR = Path("data/combined/eval")
TEST_DIR = Path("data/test")
BATCH_SIZE = 128
EPOCHS = 2

# =====================================================
# Load model
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
).to(device)


model.gradient_checkpointing_enable()

model.max_seq_length = 512


# =====================================================
# Build training dataset
# =====================================================

eval_queries = load_dataset("json", data_files=str(EVAL_DIR / "queries.jsonl"))["train"]

eval_queries = {row["id"]: row["value"] for row in eval_queries}

eval_corpus = load_dataset("json", data_files=str(EVAL_DIR / "corpus.jsonl"))["train"]

eval_corpus = {row["id"]: row["value"] for row in eval_corpus}


eval_relevant_docs = load_dataset(
    "json", data_files=str(EVAL_DIR / "relevant_docs.jsonl")
)["train"]

eval_relevant_docs = {row["id"]: set(row["value"]) for row in eval_relevant_docs}

eval_pairs = load_dataset("json", data_files=str(EVAL_DIR / "combined_pairs.jsonl"))[
    "train"
]

train_dataset = load_dataset("json", data_files=TRAIN_DIR)["train"]

val_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name="validation-document-linking",
    accuracy_at_k=[1, 5, 10],
    mrr_at_k=[1, 5, 10],
    ndcg_at_k=[1, 5, 10],
    show_progress_bar=True,
)


# =====================================================
# Loss
# =====================================================

loss = losses.CachedMultipleNegativesRankingLoss(
    model,
)


# =====================================================
# Optimizer
# =====================================================

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)


# =====================================================
# Training args
# =====================================================

training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.05,
    num_train_epochs=EPOCHS,
    logging_steps=max(1, len(train_dataset) // (BATCH_SIZE * 100)),
    logging_strategy="steps",
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    metric_for_best_model="eval_validation-document-linking_cosine_mrr@10",
    greater_is_better=True,
    report_to=["tensorboard"],
)

# =====================================================
# Trainer
# =====================================================

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_pairs,
    evaluator=val_evaluator,
    loss=loss,
    optimizers=(
        optimizer,
        None,
    ),
)


# =====================================================
# Train
# =====================================================

trainer.train()


# =====================================================
# Final test evaluation
# =====================================================

test_queries = load_dataset("json", data_files=str(TEST_DIR / "queries.jsonl"))["train"]

test_queries = {row["id"]: row["value"] for row in test_queries}

test_corpus = load_dataset("json", data_files=str(TEST_DIR / "corpus.jsonl"))["train"]

test_corpus = {row["id"]: row["value"] for row in test_corpus}

test_relevant_docs = load_dataset(
    "json", data_files=str(TEST_DIR / "relevant_docs.jsonl")
)["train"]

test_relevant_docs = {row["id"]: set(row["value"]) for row in test_relevant_docs}

test_evaluator = InformationRetrievalEvaluator(
    queries=test_queries,
    corpus=test_corpus,
    relevant_docs=test_relevant_docs,
    name="final-test-document-linking",
    accuracy_at_k=[1, 5, 10],
    mrr_at_k=[1, 5, 10],
    ndcg_at_k=[1, 5, 10],
    show_progress_bar=True,
)


results = test_evaluator(
    model,
)


print(results)
