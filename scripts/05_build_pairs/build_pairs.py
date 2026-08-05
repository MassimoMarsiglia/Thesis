import json
from pathlib import Path

from datasets import Dataset, load_dataset
from pydantic import BaseModel

INPUT_TRAIN = "data/training_pairs_v2.jsonl"
INPUT_TEST = "data/test_pairs.jsonl"
INPUT_EVAL = "data/eval_pairs.jsonl"
INPUT_CORPUS = "data/docs.jsonl"
INPUT_ENRICHED = "data/enriched_training_pairs_sbert_v2.jsonl"
INPUT_ENRICHED_TEST = "data/enriched_test_pairs_sbert_v2.jsonl"

OUTPUT_TRAIN = "data/formatted_training_pairs.jsonl"
OUTPUT_TEST = "data/test"
OUTPUT_EVAL = "data/eval"
OUTPUT_ENRICHED = "data/enriched"
OUTPUT_ENRICHED_TEST = "data/enriched/test"


class DocumentReference(BaseModel):
    description: str
    ref_title: str
    relationship_type: str
    source_build_id: str


class DocumentRecord(BaseModel):
    build_id: str
    short_title: str
    aliases: list[str] | None = []
    title: str
    doc_type: str


def format_reference(ref: DocumentReference) -> str:
    return f"""[REFERENCE DOCUMENT]
Title: {ref.ref_title}

[LINK TYPE]
{ref.relationship_type}

[LINK DESCRIPTION]
{ref.description}
"""


def format_record(record: DocumentRecord) -> str:
    aliases = record.aliases or []
    if isinstance(aliases, str):
        aliases = json.loads(aliases)

    return f"""[TARGET DOCUMENT]
Title: {record.title}

[DOCUMENT METADATA]
Short title: {record.short_title}
Document type: {record.doc_type}

[ALIASES]
{chr(10).join(f"- {a}" for a in aliases)}
"""


def format_link_pair(ref: DocumentReference, record: DocumentRecord):
    positive = format_record(record)
    anchor = format_reference(ref)

    return anchor.strip(), positive.strip()


def transform_pair(example):
    data = example["json"]

    if data["positive"].get("aliases") in (None, "None", ""):
        data["positive"]["aliases"] = []

    ref = DocumentReference(**data["reference"])
    record = DocumentRecord(**data["positive"])

    anchor, positive = format_link_pair(ref, record)

    return {
        "anchor": anchor,
        "positive": positive,
        "anchor_doc_id": data["reference"]["source_build_id"],
        "target_doc_id": data["positive"]["build_id"],
        "link_type": data["reference"]["relationship_type"],
    }


def build_ir_eval_dataset(
    eval_pairs,
    corpus_records: list[DocumentRecord],
):
    queries = {}
    corpus = {}
    relevant_docs = {}

    # Build full corpus
    for record in corpus_records:
        if record.aliases in (None, "None", ""):
            record.aliases = []

        corpus[str(record.build_id)] = format_record(record)

    # Build queries + relevance
    for example in eval_pairs:
        data = example["json"]

        ref = DocumentReference(**data["reference"])
        record = DocumentRecord(**data["positive"])

        if record.aliases in (None, "None", ""):
            record.aliases = []

        anchor_id = str(ref.source_build_id)
        target_id = str(record.build_id)

        queries[anchor_id] = format_reference(ref)

        if anchor_id not in relevant_docs:
            relevant_docs[anchor_id] = set()

        relevant_docs[anchor_id].add(target_id)

    return queries, corpus, relevant_docs


training_pairs = load_dataset("json", data_files=INPUT_TRAIN)

training_pairs = training_pairs.map(transform_pair)

training_pairs["train"].to_json(
    OUTPUT_TRAIN,
    lines=True,
)

docs = load_dataset(
    "json",
    data_files=INPUT_CORPUS,
)["train"]

if "doc_url" in docs.column_names:
    docs = docs.remove_columns(["doc_url"])

corpus_records = [DocumentRecord(**doc) for doc in docs]
test_pairs = load_dataset("json", data_files=INPUT_TEST)["train"]

test_queries, corpus, test_relevant_docs = build_ir_eval_dataset(
    test_pairs, corpus_records=corpus_records
)

Dataset.from_list([{"id": k, "value": v} for k, v in test_queries.items()]).to_json(
    Path(OUTPUT_TEST) / "queries.jsonl",
    lines=True,
)


Dataset.from_list([{"id": k, "value": v} for k, v in corpus.items()]).to_json(
    Path(OUTPUT_TEST) / "corpus.jsonl",
    lines=True,
)


Dataset.from_list(
    [{"id": k, "value": list(v)} for k, v in test_relevant_docs.items()]
).to_json(
    Path(OUTPUT_TEST) / "relevant_docs.jsonl",
    lines=True,
)

eval_pairs = load_dataset("json", data_files=INPUT_EVAL)["train"]

eval_queries, corpus, eval_relevant_docs = build_ir_eval_dataset(
    eval_pairs, corpus_records=corpus_records
)

Dataset.from_list([{"id": k, "value": v} for k, v in eval_queries.items()]).to_json(
    Path(OUTPUT_EVAL) / "queries.jsonl",
    lines=True,
)


Dataset.from_list([{"id": k, "value": v} for k, v in corpus.items()]).to_json(
    Path(OUTPUT_EVAL) / "corpus.jsonl",
    lines=True,
)


Dataset.from_list(
    [{"id": k, "value": list(v)} for k, v in eval_relevant_docs.items()]
).to_json(
    Path(OUTPUT_EVAL) / "relevant_docs.jsonl",
    lines=True,
)

enriched_pairs = load_dataset("json", data_files=INPUT_ENRICHED)["train"]

enriched_pairs = enriched_pairs.map(transform_pair)

enriched_pairs.to_json(Path(OUTPUT_ENRICHED) / "pairs.jsonl", lines=True)

enriched_pairs = load_dataset("json", data_files=INPUT_ENRICHED)["train"]

enriched_queries, corpus, enriched_relevant_docs = build_ir_eval_dataset(
    enriched_pairs, corpus_records=corpus_records
)

Dataset.from_list([{"id": k, "value": v} for k, v in enriched_queries.items()]).to_json(
    Path(OUTPUT_ENRICHED) / "queries.jsonl",
    lines=True,
)


Dataset.from_list([{"id": k, "value": v} for k, v in corpus.items()]).to_json(
    Path(OUTPUT_ENRICHED) / "corpus.jsonl",
    lines=True,
)


Dataset.from_list(
    [{"id": k, "value": list(v)} for k, v in enriched_relevant_docs.items()]
).to_json(
    Path(OUTPUT_ENRICHED) / "relevant_docs.jsonl",
    lines=True,
)

enriched_benchmark_pairs = enriched_pairs.map(transform_pair)

enriched_benchmark_pairs.to_json(Path(OUTPUT_ENRICHED) / "pairs.jsonl", lines=True)

enriched_test_pairs = load_dataset("json", data_files=INPUT_ENRICHED_TEST)["train"]

enriched_test_pairs = enriched_test_pairs.map(transform_pair)

enriched_test_pairs.to_json(Path(OUTPUT_ENRICHED_TEST) / "pairs.jsonl", lines=True)

enriched_test_pairs = load_dataset("json", data_files=INPUT_ENRICHED_TEST)["train"]

enriched_test_queries, corpus, enriched_test_relevant_docs = build_ir_eval_dataset(
    enriched_test_pairs, corpus_records=corpus_records
)

Dataset.from_list(
    [{"id": k, "value": v} for k, v in enriched_test_queries.items()]
).to_json(
    Path(OUTPUT_ENRICHED_TEST) / "queries.jsonl",
    lines=True,
)


Dataset.from_list([{"id": k, "value": v} for k, v in corpus.items()]).to_json(
    Path(OUTPUT_ENRICHED_TEST) / "corpus.jsonl",
    lines=True,
)


Dataset.from_list(
    [{"id": k, "value": list(v)} for k, v in enriched_test_relevant_docs.items()]
).to_json(
    Path(OUTPUT_ENRICHED_TEST) / "relevant_docs.jsonl",
    lines=True,
)
