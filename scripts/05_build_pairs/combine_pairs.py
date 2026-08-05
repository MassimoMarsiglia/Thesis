from collections import defaultdict
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

INPUT_TRAIN = "data/formatted_training_pairs.jsonl"
INPUT_EVAL_PAIRS = "data/eval/pairs.jsonl"
INPUT_EVAL_QUERIES = "data/eval/queries.jsonl"
INPUT_EVAL_RELEVANT_DOCS = "data/eval/relevant_docs.jsonl"

INPUT_ENRICHED_TRAIN = "data/enriched/pairs.jsonl"
INPUT_ENRICHED_EVAL_PAIRS = "data/enriched/test/pairs.jsonl"
INPUT_ENRICHED_EVAL_QUERIES = "data/enriched/test/queries.jsonl"
INPUT_ENRICHED_EVAL_RELEVANT_DOCS = "data/enriched/test/relevant_docs.jsonl"


OUTPUT_TRAIN = "data/combined/combined_training_pairs.jsonl"
OUTPUT_EVAL = "data/combined/eval/combined_pairs.jsonl"
OUTPUT_ENRICHED = "data/enriched/combined_pairs.jsonl"
OUTPUT_COMBINED = "data/combined"


def load_and_merge_related_docs(paths):
    merged = defaultdict(list)

    for path in paths:
        ds = load_dataset(
            "json",
            data_files=path,
        )["train"]

        for row in ds:
            merged[row["id"]].extend(row["value"])

    # optional: remove duplicate document ids per query
    merged = {k: list(set(v)) for k, v in merged.items()}

    return Dataset.from_dict(
        {
            "id": list(merged.keys()),
            "value": list(merged.values()),
        }
    )


def format_training_dataset(ds: Dataset) -> Dataset:
    return ds.select_columns(
        [
            "anchor",
            "positive",
        ]
    ).rename_columns(
        {
            "anchor": "sentence_1",
            "positive": "sentence_2",
        }
    )


datasets = []

for path in [
    INPUT_TRAIN,
    INPUT_ENRICHED_TRAIN,
    # add more files here
]:
    ds = load_dataset(
        "json",
        data_files=path,
    )["train"]

    ds = format_training_dataset(ds)

    datasets.append(ds)

dataset = concatenate_datasets(datasets)

dataset.to_json(
    OUTPUT_TRAIN,
    lines=True,
)

datasets = []

for path in [
    INPUT_EVAL_PAIRS,
    INPUT_ENRICHED_EVAL_PAIRS,
    # add more files here
]:
    ds = load_dataset(
        "json",
        data_files=path,
    )["train"]

    ds = format_training_dataset(ds)

    datasets.append(ds)

dataset = concatenate_datasets(datasets)

dataset.to_json(
    OUTPUT_EVAL,
    lines=True,
)

datasets = []

for path in [
    INPUT_ENRICHED_TRAIN
    # add more files here
]:
    ds = load_dataset(
        "json",
        data_files=path,
    )["train"]

    ds = format_training_dataset(ds)

    datasets.append(ds)

dataset = concatenate_datasets(datasets)

dataset.to_json(
    OUTPUT_ENRICHED,
    lines=True,
)

datasets = []

for path in [
    INPUT_EVAL_QUERIES,
    INPUT_ENRICHED_EVAL_QUERIES,
    # add more files here
]:
    ds = load_dataset(
        "json",
        data_files=path,
    )["train"]

    datasets.append(ds)

dataset = concatenate_datasets(datasets)

dataset.to_json(
    Path(OUTPUT_COMBINED) / "queries.jsonl",
    lines=True,
)

datasets = []

related_docs = load_and_merge_related_docs(
    [
        INPUT_EVAL_RELEVANT_DOCS,
        INPUT_ENRICHED_EVAL_RELEVANT_DOCS,
    ]
)

related_docs.to_json(
    Path(OUTPUT_COMBINED) / "relevant_docs.jsonl",
    lines=True,
)
