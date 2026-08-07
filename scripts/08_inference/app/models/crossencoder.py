from sentence_transformers import CrossEncoder


class RerankerModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):

        self.model = CrossEncoder(
            model_path,
            device=device,
        )

    def score(
        self,
        query: str,
        candidates: list[str],
        batch_size: int = 32,
    ):

        pairs = [
            (
                query,
                candidate,
            )
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        return scores.tolist()
