from __future__ import annotations

import asyncio
import csv
import json
from collections import Counter
from dataclasses import dataclass

import aiohttp

from app.facets import FACETS_CSV, load_facets
from app.evaluator import chunked, parse_json_object
from app.models import Facet


@dataclass
class FacetCategoryClient:
    base_url: str
    model_name: str
    timeout_seconds: int = 300
    session: aiohttp.ClientSession | None = None

    async def suggest_categories(
        self,
        facets: list[Facet],
        max_categories: int,
    ) -> list[str]:
        if self.session:
            return await self._suggest_categories(facets, max_categories)

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                return await self._suggest_categories(facets, max_categories)
            finally:
                self.session = None

    async def categorize(
        self,
        facets: list[Facet],
        allowed_categories: list[str],
    ) -> list[dict[str, str]]:
        if self.session:
            return await self._categorize(facets, allowed_categories)

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            try:
                return await self._categorize(facets, allowed_categories)
            finally:
                self.session = None

    async def _suggest_categories(
        self,
        facets: list[Facet],
        max_categories: int,
    ) -> list[str]:
        facet_lines = "\n".join(f"{facet.facet_id}: {facet.name}" for facet in facets)
        prompt = f"""Create a compact taxonomy for these facets.
Return only valid JSON with this exact shape:
{{"categories":["emotion","safety"]}}

Rules:
- Return at most {max_categories} categories.
- Use broad reusable labels, not one category per facet.
- Use short lowercase labels with underscores.
- Include "unclear" as one category.

Facets:
{facet_lines}
"""
        parsed = await self._generate_json(prompt, max(256, max_categories * 24))
        categories = parsed.get("categories", []) if isinstance(parsed, dict) else []
        return normalize_categories(categories, max_categories)

    async def _categorize(
        self,
        facets: list[Facet],
        allowed_categories: list[str],
    ) -> list[dict[str, str]]:
        facet_lines = "\n".join(f"{facet.facet_id}: {facet.name}" for facet in facets)
        allowed = ", ".join(allowed_categories)
        prompt = f"""Assign one category to each facet.
Return only valid JSON with this exact shape:
{{"facets":[{{"facet_id":"F0001","category":"emotion"}}]}}

Allowed categories:
{allowed}

Rules:
- Use only the allowed categories.
- Choose the closest broad category.
- Use "unclear" only when no allowed category fits.
- Return every listed facet exactly once.

Facets:
{facet_lines}
"""
        parsed = await self._generate_json(prompt, max(256, len(facets) * 32))
        items = parsed.get("facets", []) if isinstance(parsed, dict) else []
        return [item for item in items if isinstance(item, dict)]

    async def _generate_json(self, prompt: str, num_predict: int) -> dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": num_predict,
            },
        }
        url = f"{self.base_url.rstrip('/')}/api/generate"
        session = self.session
        if not session:
            raise RuntimeError("Ollama facet processing session is not initialized.")
        try:
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Ollama returned HTTP {response.status}: {detail}")
                data = await response.json()
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Ollama facet processing timed out.") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Ollama facet processing failed: {exc}") from exc

        return parse_json_object(data.get("response", "{}"))


def normalize_category(value: object) -> str:
    category = str(value or "unclear").strip().lower()
    category = "_".join(category.split())
    category = "".join(char for char in category if char.isalnum() or char == "_")
    return category[:80] or "unclear"


def normalize_categories(values: list[object], max_categories: int) -> list[str]:
    counts: Counter[str] = Counter()
    for value in values:
        category = normalize_category(value)
        counts[category] += 1

    categories = [
        category
        for category, _ in counts.most_common()
        if category != "unclear"
    ]
    categories = categories[: max_categories - 1]
    categories.append("unclear")
    return categories


async def process_facets_with_ollama(
    client: FacetCategoryClient,
    batch_size: int,
    max_categories: int,
) -> tuple[list[Facet], str]:
    max_categories = max(2, max_categories)
    facets = load_facets()
    suggestions: list[str] = []
    timeout = aiohttp.ClientTimeout(total=client.timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        client.session = session
        try:
            for batch in chunked(facets, batch_size):
                suggestions.extend(await client.suggest_categories(batch, max_categories))
            allowed_categories = normalize_categories(suggestions, max_categories)

            categories_by_id: dict[str, str] = {}
            for batch in chunked(facets, batch_size):
                for item in await client.categorize(batch, allowed_categories):
                    category = normalize_category(item.get("category", "unclear"))
                    if category not in allowed_categories:
                        category = "unclear"
                    categories_by_id[str(item.get("facet_id"))] = category
        finally:
            client.session = None

    records = []
    for facet in facets:
        records.append(
            {
                "facet_id": facet.facet_id,
                "name": facet.name,
                "category": categories_by_id.get(facet.facet_id, "unclear"),
            }
        )

    with FACETS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["facet_id", "name", "category"])
        writer.writeheader()
        writer.writerows(records)

    load_facets.cache_clear()
    return load_facets(), str(FACETS_CSV)
