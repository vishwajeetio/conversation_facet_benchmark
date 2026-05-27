from __future__ import annotations

import asyncio
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.models import ConversationTurn, Facet, FacetScore


SCORE_SCALE = [-2, -1, 0, 1, 2]


class LLMClient(Protocol):
    model_name: str

    async def score_batch(
        self,
        conversation_id: str,
        turn_index: int,
        turn: ConversationTurn,
        facets: list[Facet],
    ) -> list[FacetScore]:
        ...


def chunked(items: list[Facet], size: int) -> list[list[Facet]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def coerce_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(-2, min(2, score))


def coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.35
    return max(0.0, min(1.0, confidence))


@dataclass
class OllamaClient:
    base_url: str
    model_name: str
    timeout_seconds: int = 300
    num_ctx: int = 4096

    async def score_batch(
        self,
        conversation_id: str,
        turn_index: int,
        turn: ConversationTurn,
        facets: list[Facet],
    ) -> list[FacetScore]:
        return await asyncio.to_thread(
            self._score_batch_sync, conversation_id, turn_index, turn, facets
        )

    def _score_batch_sync(
        self,
        conversation_id: str,
        turn_index: int,
        turn: ConversationTurn,
        facets: list[Facet],
    ) -> list[FacetScore]:
        prompt = build_prompt(turn_index, turn, facets)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
                "num_predict": max(256, len(facets) * 96),
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                "Ollama timed out while scoring a facet batch. "
                "Try a smaller FACET_BATCH_SIZE or a smaller model."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                "Ollama timed out while scoring a facet batch. "
                "Try a smaller FACET_BATCH_SIZE or a smaller model."
            ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        raw_response = data.get("response", "{}")
        try:
            parsed = parse_json_object(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned malformed JSON for a facet batch. "
                "Retry or reduce FACET_BATCH_SIZE."
            ) from exc
        items = parsed.get("scores", []) if isinstance(parsed, dict) else []
        return normalize_scores(conversation_id, turn_index, turn, facets, items)


def build_prompt(turn_index: int, turn: ConversationTurn, facets: list[Facet]) -> str:
    facet_lines = "\n".join(
        f"{facet.facet_id}: {facet.name} ({facet.category})"
        for facet in facets
    )
    return f"""Score one conversation turn against each listed facet.
Return only valid JSON with this exact shape:
{{"scores":[{{"facet_id":"F0001","score":0,"confidence":0.72,"rationale":"short evidence"}}]}}

Rules:
- Score every listed facet exactly once.
- Use only these ordered integer scores: -2, -1, 0, 1, 2.
- -2 means strong evidence against; -1 weak against; 0 no clear evidence; 1 weak for; 2 strong for.
- Confidence is 0 to 1.
- Prefer score 0 unless the turn gives evidence.
- Keep each rationale under 12 words.

Turn index: {turn_index}
Speaker: {turn.speaker}
Text:
{turn.text}

Facets:
{facet_lines}
"""


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_scores(
    conversation_id: str,
    turn_index: int,
    turn: ConversationTurn,
    facets: list[Facet],
    items: list[dict],
) -> list[FacetScore]:
    by_id = {str(item.get("facet_id")): item for item in items if isinstance(item, dict)}
    scores: list[FacetScore] = []
    for facet in facets:
        item = by_id.get(facet.facet_id, {})
        scores.append(
            FacetScore(
                conversation_id=conversation_id,
                turn_index=turn_index,
                speaker=turn.speaker,
                facet_id=facet.facet_id,
                facet_name=facet.name,
                score=coerce_score(item.get("score", 0)),
                confidence=coerce_confidence(item.get("confidence", 0.25)),
                rationale=str(item.get("rationale", "No model rationale returned."))[:500],
            )
        )
    return scores


def make_client() -> LLMClient:
    return OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    )


async def evaluate_conversation(
    conversation_id: str,
    turns: list[ConversationTurn],
    facets: list[Facet],
    client: LLMClient | None = None,
    batch_size: int | None = None,
) -> list[FacetScore]:
    client = client or make_client()
    batch_size = batch_size or int(os.getenv("FACET_BATCH_SIZE", "8"))
    all_scores: list[FacetScore] = []
    for turn_index, turn in enumerate(turns):
        for batch in chunked(facets, batch_size):
            all_scores.extend(
                await client.score_batch(conversation_id, turn_index, turn, batch)
            )
    return all_scores
