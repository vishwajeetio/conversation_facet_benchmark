from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

import aiohttp

from app.models import ConversationTurn, Facet, FacetScore


SCORE_SCALE = [-2, -1, 0, 1, 2]
LOGGER = logging.getLogger(__name__)


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
    session: aiohttp.ClientSession | None = None

    async def score_batch(
        self,
        conversation_id: str,
        turn_index: int,
        turn: ConversationTurn,
        facets: list[Facet],
    ) -> list[FacetScore]:
        if self.session:
            return await self._score_batch_async(
                self.session,
                conversation_id,
                turn_index,
                turn,
                facets,
            )

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await self._score_batch_async(
                session,
                conversation_id,
                turn_index,
                turn,
                facets,
            )

    async def _score_batch_async(
        self,
        session: aiohttp.ClientSession,
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
        url = f"{self.base_url.rstrip('/')}/api/generate"
        try:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Ollama returned HTTP {response.status}: {detail}")
                data = await response.json()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Ollama timed out while scoring a facet batch. "
                "Try a smaller FACET_BATCH_SIZE or a smaller model."
            ) from exc
        except aiohttp.ClientError as exc:
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
    max_concurrency: int | None = None,
) -> list[FacetScore]:
    client = client or make_client()
    batch_size = batch_size or int(os.getenv("FACET_BATCH_SIZE", "8"))
    max_concurrency = max_concurrency or int(os.getenv("EVALUATION_MAX_CONCURRENCY", "2"))
    max_concurrency = max(1, max_concurrency)
    semaphore = asyncio.Semaphore(max_concurrency)
    jobs: list[tuple[int, int, ConversationTurn, list[Facet]]] = []
    for turn_index, turn in enumerate(turns):
        for batch_index, batch in enumerate(chunked(facets, batch_size)):
            jobs.append((turn_index, batch_index, turn, batch))

    LOGGER.info(
        "Starting evaluation conversation_id=%s turns=%s facets=%s jobs=%s batch_size=%s concurrency=%s",
        conversation_id,
        len(turns),
        len(facets),
        len(jobs),
        batch_size,
        max_concurrency,
    )
    if len(jobs) < max_concurrency:
        LOGGER.warning(
            "Evaluation has fewer jobs (%s) than concurrency (%s); increase facet limit, turns, or lower batch size to keep Ollama busy.",
            len(jobs),
            max_concurrency,
        )

    async def score_job(
        turn_index: int,
        batch_index: int,
        turn: ConversationTurn,
        batch: list[Facet],
    ) -> tuple[int, int, list[FacetScore]]:
        async with semaphore:
            LOGGER.info(
                "Scoring turn=%s batch=%s facets=%s",
                turn_index,
                batch_index,
                len(batch),
            )
            scores = await client.score_batch(conversation_id, turn_index, turn, batch)
            return turn_index, batch_index, scores

    if isinstance(client, OllamaClient):
        timeout = aiohttp.ClientTimeout(total=client.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            client.session = session
            try:
                completed = await asyncio.gather(
                    *(
                        score_job(turn_index, batch_index, turn, batch)
                        for turn_index, batch_index, turn, batch in jobs
                    )
                )
            finally:
                client.session = None
    else:
        completed = await asyncio.gather(
            *(
                score_job(turn_index, batch_index, turn, batch)
                for turn_index, batch_index, turn, batch in jobs
            )
        )
    completed.sort(key=lambda item: (item[0], item[1]))
    return [score for _, _, scores in completed for score in scores]
