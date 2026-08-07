import json

from index import FaissIndex
from models.biencoder import EmbeddingModel
from models.crossencoder import RerankerModel
from pipeline import LinkerPipeline
from schemas import ReferenceDocument
from sklearn.metrics import ndcg_score
from tqdm import tqdm

TEST_PATH = "data/test_pairs.jsonl"


def load_jsonl(path):

    with open(path) as f:
        return [json.loads(line) for line in f]


# -------------------------
# Load models
# -------------------------

embedder = EmbeddingModel(
    "artifacts/models/embeddinggemma-document-linker/checkpoint-170"
)


reranker = RerankerModel(
    "artifacts/models/embeddinggemma-document-link-reranker/checkpoint-4074"
)


index = FaissIndex(
    "artifacts/index.faiss",
    "data/test/corpus.jsonl",
    embedder,
)

index.init()


linker = LinkerPipeline(
    embedder,
    reranker,
    index,
)


# -------------------------
# Load evaluation data
# -------------------------

dataset = load_jsonl(TEST_PATH)


recall_hits = 0
accuracy_hits = 0

mrr_scores = []
ndcg_scores = []


# -------------------------
# Evaluation loop
# -------------------------

for sample in tqdm(dataset):
    sample = sample["json"]
    reference = sample["reference"]
    positive = sample["positive"]

    target_id = positive["build_id"]

    document = ReferenceDocument(
        title=reference["ref_title"],
        description=reference["description"],
        relationship_type=reference["relationship_type"],
    )

    results = linker.predict(
        document,
        top_k=50,
    )

    ranked_ids = [candidate["id"] for candidate in results]

    # -------------------------
    # Recall@50
    # -------------------------

    if target_id in ranked_ids:
        recall_hits += 1

    # -------------------------
    # Accuracy@1
    # -------------------------

    if ranked_ids[0] == target_id:
        accuracy_hits += 1

    # -------------------------
    # MRR@50
    # -------------------------

    if target_id in ranked_ids:
        rank = ranked_ids.index(target_id) + 1

        mrr_scores.append(1 / rank)

    else:
        mrr_scores.append(0)

    # -------------------------
    # NDCG@50
    # -------------------------

    relevance = [1 if candidate_id == target_id else 0 for candidate_id in ranked_ids]

    ndcg_scores.append(ndcg_score([relevance], [list(range(len(relevance), 0, -1))]))


# -------------------------
# Final results
# -------------------------

total = len(dataset)


print("\nEvaluation Results")
print("--------------------------")

print(f"Recall@50: {recall_hits / total:.4f}")


print(f"Accuracy@1: {accuracy_hits / total:.4f}")


print(f"MRR@50: {sum(mrr_scores) / total:.4f}")


print(f"NDCG@50: {sum(ndcg_scores) / total:.4f}")
