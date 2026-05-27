from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.facet_processing import FacetCategoryClient, process_facets_with_ollama


async def main() -> None:
    client = FacetCategoryClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model_name=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
    )
    facets, path = await process_facets_with_ollama(
        client,
        batch_size=int(os.getenv("FACET_PROCESS_BATCH_SIZE", "30")),
        max_categories=int(os.getenv("FACET_MAX_CATEGORIES", "12")),
    )
    categories = sorted({facet.category for facet in facets})
    print(f"Processed {len(facets)} facets with {client.model_name}")
    print(f"Saved: {path}")
    print(f"Categories: {', '.join(categories)}")


if __name__ == "__main__":
    asyncio.run(main())
