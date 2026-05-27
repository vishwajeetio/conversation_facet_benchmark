from __future__ import annotations

from pydantic import BaseModel, Field, conint


class ConversationTurn(BaseModel):
    speaker: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=5000)


class EvaluationRequest(BaseModel):
    conversation_id: str = Field(default="ad-hoc", max_length=120)
    turns: list[ConversationTurn] = Field(min_length=1, max_length=100)
    facet_ids: list[str] | None = None


class Facet(BaseModel):
    facet_id: str
    name: str
    category: str


class FacetScore(BaseModel):
    conversation_id: str
    turn_index: int
    speaker: str
    facet_id: str
    facet_name: str
    score: conint(ge=-2, le=2)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=500)


class EvaluationResponse(BaseModel):
    conversation_id: str
    run_id: str
    score_scale: list[int]
    facet_count: int
    turn_count: int
    model: str
    saved_path: str
    results: list[FacetScore]
