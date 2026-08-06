from datasets import Dataset, load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)

INPUT = "data/train/mined_training_pairs_v2.jsonl"
OUTPUT = "artifacts/models/embeddinggemma-document-link-reranker"
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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

model = CrossEncoder(MODEL, num_labels=1)


args = CrossEncoderTrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=50,
    save_strategy="epoch",
)

loss = BinaryCrossEntropyLoss(model)


trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_pairs,
    loss=loss,
)


trainer.train()
