class LinkerPipeline:
    def __init__(
        self,
        embedding_model,
        reranker_model,
        index,
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.index = index

    def format_reference(
        self,
        document,
    ):

        return f"""
[REFERENCE DOCUMENT]

Title: {document.title}

[LINK TYPE]

{document.relationship_type}

[LINK DESCRIPTION]

{document.description}
""".strip()

    def predict(
        self,
        document,
        top_k=50,
    ):

        reference_text = self.format_reference(document)

        query_embedding = self.embedding_model.encode_one(reference_text)

        candidates = self.index.search(
            query_embedding,
            top_k,
        )

        candidate_texts = [c["value"] for c in candidates]

        scores = self.reranker_model.score(
            reference_text,
            candidate_texts,
        )

        for candidate, score in zip(
            candidates,
            scores,
        ):
            candidate["reranker_score"] = float(score)

        return sorted(
            candidates,
            key=lambda x: x["reranker_score"],
            reverse=True,
        )
