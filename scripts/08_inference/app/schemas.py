from pydantic import BaseModel


class ReferenceDocument(BaseModel):
    title: str
    description: str
    relationship_type: str


class CandidateResult(BaseModel):
    id: str
    value: str
    embedding_score: float
    reranker_score: float


class LinkResponse(BaseModel):
    candidates: list[CandidateResult]
