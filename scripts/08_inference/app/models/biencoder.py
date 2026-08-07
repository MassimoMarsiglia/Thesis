import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        self.model = SentenceTransformer(
            model_path,
            device=device,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = False,
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=convert_to_numpy,
        )

        return np.asarray(
            embeddings,
            dtype="float32",
        )

    def encode_one(
        self,
        text: str,
    ) -> np.ndarray:

        return self.encode([text], convert_to_numpy=True)[0]
