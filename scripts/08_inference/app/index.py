import json
import os

import faiss
import numpy as np


class FaissIndex:
    def __init__(
        self,
        index_path: str,
        docs_path: str,
        embedder,
    ):
        self.index_path = index_path
        self.docs_path = docs_path
        self.embedder = embedder

        self.index = None
        self.documents = []

    def init(self):

        if os.path.exists(self.index_path):
            self.load()

        else:
            self.build()

    def load(self):

        self.index = faiss.read_index(self.index_path)

        with open(self.docs_path) as f:
            self.documents = [json.loads(x) for x in f]

    def save(
        self,
        index,
    ):

        directory = os.path.dirname(self.index_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        faiss.write_index(
            index,
            self.index_path,
        )

    def build(self):

        with open(self.docs_path) as f:
            self.documents = [json.loads(x) for x in f]

        texts = [doc["value"] for doc in self.documents]

        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
        )

        dimension = embeddings.shape[1]

        base_index = faiss.IndexFlatIP(dimension)

        self.index = faiss.IndexIDMap(base_index)

        # FAISS requires int64 IDs
        ids = np.arange(
            len(self.documents),
            dtype="int64",
        )

        self.index.add_with_ids(
            embeddings,
            ids,
        )

        self.save(self.index)

    def search(
        self,
        embedding,
        k: int,
    ):

        if self.index is None:
            raise RuntimeError("FAISS index has not been initialized")

        scores, ids = self.index.search(
            embedding[None, :],
            k,
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0],
        ):
            if idx == -1:
                continue

            results.append(
                {
                    **self.documents[idx],
                    "embedding_score": float(score),
                }
            )

        return results
