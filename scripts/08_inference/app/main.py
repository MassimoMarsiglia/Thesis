from fastapi import FastAPI
from index import FaissIndex
from models.biencoder import EmbeddingModel
from models.crossencoder import RerankerModel
from pipeline import LinkerPipeline
from schemas import LinkResponse, ReferenceDocument

app = FastAPI()


embedder = EmbeddingModel(
    "artifacts/models/embeddinggemma-document-linker/checkpoint-170"
)


reranker = RerankerModel(
    "artifacts/models/embeddinggemma-document-link-reranker/checkpoint-4074"
)


index = FaissIndex("artifacts/index.faiss", "data/test/corpus.jsonl", embedder=embedder)

index.init()


linker = LinkerPipeline(
    embedder,
    reranker,
    index,
)


@app.post(
    "/link",
    response_model=LinkResponse,
)
def link(
    document: ReferenceDocument,
):

    results = linker.predict(
        document,
        top_k=50,
    )

    print(f"hey {results[0]}")
    return {"candidates": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
