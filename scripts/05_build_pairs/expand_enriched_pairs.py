import json

from datasets import Dataset, load_dataset

INPUT_TRAIN = "data/enriched_training_pairs_v2.jsonl"
INPUT_TEST = "data/enriched_test_pairs_v2.jsonl"

OUTPUT_TRAIN = "data/enriched_training_pairs_sbert_v2.jsonl"
OUTPUT_TEST = "data/enriched_test_pairs_sbert_v2.jsonl"


def expand_generated(example):
    data = example["source"]["json"]

    outputs = []

    aliases = data["positive"].get("aliases", [])
    if isinstance(aliases, str):
        aliases = [] if aliases in ("", "None") else json.loads(aliases)

    positive = {
        **data["positive"],
        "aliases": aliases,
    }

    for generated in example["generated"]:
        new_example = {
            "json": {
                "reference": {
                    "source_build_id": data["reference"]["source_build_id"],
                    "ref_title": generated["title"],
                    "relationship_type": data["reference"]["relationship_type"],
                    "description": generated["description"],
                },
                "positive": positive,
                "strategy": "synthetic",
            }
        }

        outputs.append(new_example)

    return outputs


dataset = load_dataset("json", data_files=INPUT_TRAIN)["train"]

expanded = []

for example in dataset:
    ex = expand_generated(example)
    expanded.extend(ex)

Dataset.from_list(expanded).to_json(
    OUTPUT_TRAIN,
    lines=True,
)

dataset = load_dataset("json", data_files=INPUT_TEST)["train"]

expanded = []

for example in dataset:
    ex = expand_generated(example)
    expanded.extend(ex)

Dataset.from_list(expanded).to_json(
    OUTPUT_TEST,
    lines=True,
)
