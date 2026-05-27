from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.models import ConversationTurn, FacetScore


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return name or "run"


def save_run(
    conversation_id: str,
    model: str,
    turns: list[ConversationTurn],
    results: list[FacetScore],
) -> tuple[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{safe_name(conversation_id)}"
    path = RESULTS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "created_at": timestamp,
        "model": model,
        "turns": [turn.model_dump() for turn in turns],
        "results": [score.model_dump() for score in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_id, path

