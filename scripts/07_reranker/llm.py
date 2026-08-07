import bitsandbytes as bnb
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CERerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)

INPUT = "data/train/mined_training_pairs_v2.jsonl"
INPUT_EVAL = "data/reranker/test/reranker_eval.jsonl"
OUTPUT = "artifacts/models/embeddinggemma-document-link-reranker"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"

dataset = load_dataset("json", data_files=INPUT)["train"]


def flatten_triplets(dataset):
    sentence_a = []
    sentence_b = []
    labels = []

    for row in dataset:
        # positive pair
        sentence_a.append(row["sentence_1"])
        sentence_b.append(row["sentence_2"])
        labels.append(1.0)

        # negative pair
        sentence_a.append(row["sentence_1"])
        sentence_b.append(row["negative"])
        labels.append(0.0)

    return Dataset.from_dict(
        {
            "sentence_1": sentence_a,
            "sentence_2": sentence_b,
            "labels": labels,
        }
    )


train_pairs = flatten_triplets(dataset)

print(train_pairs[0])
print(train_pairs[1])
print("Total pairs:", len(train_pairs))

train_pairs = train_pairs.select_columns(
    [
        "sentence_1",
        "sentence_2",
        "labels",
    ]
)

print(train_pairs.column_names)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CrossEncoder(
    MODEL_NAME,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    max_length=768,
).to(device)

# model.gradient_checkpointing_enable()

eval_data = load_dataset("json", data_files=INPUT_EVAL)["train"]

evaluator = CERerankingEvaluator(
    samples=list(eval_data),
    at_k=1,
    name="reranker-test",
    batch_size=16,
    show_progress_bar=True,
    write_csv=True,
)


args = CrossEncoderTrainingArguments(
    output_dir=OUTPUT,
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=50,
    bf16=True,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    metric_for_best_model="eval_reranker-test_ndcg@1",
    report_to=["tensorboard"],
)

loss = BinaryCrossEntropyLoss(model)

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
)


trainer = CrossEncoderTrainer(
    optimizers=(
        optimizer,
        None,
    ),
    model=model,
    evaluator=evaluator,
    args=args,
    train_dataset=train_pairs,
    loss=loss,
)


# trainer.train()


result = evaluator(model)

print(result)
