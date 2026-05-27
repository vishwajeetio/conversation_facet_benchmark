from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from app.models import Facet


ROOT = Path(__file__).resolve().parents[1]
RAW_FACETS_CSV = ROOT / "data" / "raw" / "facets_assignment.csv"
FACETS_CSV = ROOT / "data" / "processed" / "facets.csv"


def clean_name(value: str) -> str:
    value = re.sub(r"^\d+\.\s*", "", value.strip())
    value = value.rstrip(":").strip()
    return re.sub(r"\s+", " ", value)


def bootstrap_facets() -> None:
    FACETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RAW_FACETS_CSV.open(newline="", encoding="utf-8-sig") as source:
        raw_names = [
            clean_name(row.get("Facets", ""))
            for row in csv.DictReader(source)
        ]
    rows = [
        {
            "facet_id": f"F{index:04d}",
            "name": name,
            "category": "unprocessed",
        }
        for index, name in enumerate([name for name in raw_names if name], start=1)
    ]
    with FACETS_CSV.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["facet_id", "name", "category"])
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=1)
def load_facets() -> list[Facet]:
    if not FACETS_CSV.exists():
        bootstrap_facets()
    with FACETS_CSV.open(newline="", encoding="utf-8") as handle:
        return [Facet.model_validate(row) for row in csv.DictReader(handle)]


def select_facets(facet_ids: list[str] | None = None) -> list[Facet]:
    facets = load_facets()
    if not facet_ids:
        return facets
    wanted = set(facet_ids)
    return [facet for facet in facets if facet.facet_id in wanted or facet.name in wanted]
