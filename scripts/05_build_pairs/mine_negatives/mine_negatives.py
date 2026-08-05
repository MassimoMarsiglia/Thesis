import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

INPUT_DIR = "data/combined/combined_training_pairs.jsonl"
OUTPUT_DIR = "data/combined/mined_training_pairs_v2.jsonl"
MODEL_NAME = "artifacts/models/embeddinggemma-document-linker/checkpoint-44"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
).to(device)

train_dataset = load_dataset("json", data_files=INPUT_DIR)["train"]

train_dataset = mine_hard_negatives(
    train_dataset,
    model=model,
    range_min=2,
    range_max=50,
    max_score=0.95,
    min_score=0.2,
    relative_margin=0.05,
    num_negatives=3,
    sampling_strategy="top",
    batch_size=128,
    use_faiss=True,
    output_scores=False,
)

print(train_dataset.column_names)

train_dataset.to_json(OUTPUT_DIR, lines=True)
