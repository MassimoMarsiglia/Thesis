import random
from collections import Counter

import pandas as pd
from datasets import load_dataset

TRAIN_OUTPUT = "data/training_pairs_v2.jsonl"
VAL_OUTPUT = "data/validation_pairs_v2.jsonl"
TEST_OUTPUT = "data/test_pairs_v2.jsonl"

# Load JSONL
dataset = load_dataset(
    "json",
    data_files="data/positive_pairs.jsonl",
)["train"]


# Extract build_id as a flat column
dataset = dataset.map(
    lambda x: {"positive_build_id": str(x["json"]["positive"]["build_id"])}
)


# Strategy distribution
strategy_counts = Counter(dataset["json"]["strategy"])

strategy_dist = pd.DataFrame(
    strategy_counts.items(),
    columns=["strategy", "count"],
).sort_values("count", ascending=False)

strategy_dist["percentage"] = (
    strategy_dist["count"] / strategy_dist["count"].sum() * 100
)

print(strategy_dist)


# Unique document groups
build_ids = list(set(dataset["positive_build_id"]))

# Deterministic shuffle
random.seed(42)
random.shuffle(build_ids)


# Split document groups: 80 / 13 / 7
train_split = int(len(build_ids) * 0.80)
val_split = int(len(build_ids) * 0.93)


train_build_ids = set(build_ids[:train_split])
val_build_ids = set(build_ids[train_split:val_split])
test_build_ids = set(build_ids[val_split:])


# Filter rows
train = dataset.filter(lambda x: x["positive_build_id"] in train_build_ids)

val = dataset.filter(lambda x: x["positive_build_id"] in val_build_ids)

test = dataset.filter(lambda x: x["positive_build_id"] in test_build_ids)


# Remove helper column
train = train.remove_columns(["positive_build_id"])
val = val.remove_columns(["positive_build_id"])
test = test.remove_columns(["positive_build_id"])


# Export JSONL
train.to_json(
    TRAIN_OUTPUT,
    lines=True,
)

val.to_json(
    VAL_OUTPUT,
    lines=True,
)

test.to_json(
    TEST_OUTPUT,
    lines=True,
)


# Distribution helper
def print_strategy_dist(name, split):
    counts = Counter(split["json"]["strategy"])

    dist = pd.DataFrame(
        counts.items(),
        columns=["strategy", "count"],
    ).sort_values("count", ascending=False)

    dist["percentage"] = dist["count"] / dist["count"].sum() * 100

    print(f"\n{name} strategy dist:")
    print(dist)


print_strategy_dist("Train", train)
print_strategy_dist("Validation", val)
print_strategy_dist("Test", test)


print(f"\nTrain: {len(train)} examples")
print(f"Validation: {len(val)} examples")
print(f"Test: {len(test)} examples")

print(f"Train builds: {len(train_build_ids)}")
print(f"Validation builds: {len(val_build_ids)}")
print(f"Test builds: {len(test_build_ids)}")
