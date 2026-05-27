from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.models import ConversationTurn, FacetScore


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"
RUN_STATUS_DIR = ROOT / "data" / "run_status"


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return name or "run"


def make_run_id(conversation_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{safe_name(conversation_id)}"


def save_run(
    conversation_id: str,
    model: str,
    turns: list[ConversationTurn],
    results: list[FacetScore],
    run_id: str | None = None,
) -> tuple[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run_id or make_run_id(conversation_id)
    path = RESULTS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "model": model,
        "turns": [turn.model_dump() for turn in turns],
        "results": [score.model_dump() for score in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_id, path


def write_run_status(
    run_id: str,
    status: str,
    message: str = "",
    result: dict | None = None,
) -> Path:
    RUN_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_STATUS_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "status": status,
        "message": message,
        "result": result,
        "updated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_run_status(run_id: str) -> dict | None:
    path = RUN_STATUS_DIR / f"{safe_name(run_id)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
