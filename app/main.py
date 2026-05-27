from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.evaluator import SCORE_SCALE, evaluate_conversation, make_client
from app.facets import load_facets, select_facets
from app.models import EvaluationJobResponse, EvaluationRequest, EvaluationResponse
from app.storage import make_run_id, read_run_status, save_run, write_run_status


app = FastAPI(
    title="Ocean Across Conversation Facet Benchmark",
    version="0.1.0",
    description="Ollama-backed benchmark for scoring conversation turns across scalable facet registries.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "facets": len(load_facets())}


@app.get("/api/facets")
def facets() -> dict:
    records = [facet.model_dump(by_alias=True) for facet in load_facets()]
    categories = sorted({record["category"] for record in records})
    return {"count": len(records), "categories": categories, "facets": records}


async def run_evaluation_job(
    run_id: str,
    payload: EvaluationRequest,
    facet_ids: list[str],
) -> None:
    selected = select_facets(facet_ids)
    client = make_client()
    try:
        results = await evaluate_conversation(
            payload.conversation_id,
            payload.turns,
            selected,
            client=client,
        )
        _, saved_path = save_run(
            payload.conversation_id,
            client.model_name,
            payload.turns,
            results,
            run_id=run_id,
        )
        response = EvaluationResponse(
            conversation_id=payload.conversation_id,
            run_id=run_id,
            score_scale=SCORE_SCALE,
            facet_count=len(selected),
            turn_count=len(payload.turns),
            model=client.model_name,
            saved_path=str(saved_path),
            results=results,
        )
        write_run_status(
            run_id,
            "completed",
            "Evaluation completed.",
            response.model_dump(mode="json"),
        )
    except Exception as exc:
        write_run_status(run_id, "failed", str(exc))


@app.post("/api/evaluate", response_model=EvaluationJobResponse)
async def evaluate(payload: EvaluationRequest) -> EvaluationJobResponse:
    selected = select_facets(payload.facet_ids)
    if not selected:
        raise HTTPException(status_code=400, detail="No matching facets found.")
    run_id = make_run_id(payload.conversation_id)
    facet_ids = [facet.facet_id for facet in selected]
    write_run_status(run_id, "running", "Evaluation is running.")
    asyncio.create_task(
        run_evaluation_job(
            run_id,
            payload,
            facet_ids,
        )
    )
    return EvaluationJobResponse(
        run_id=run_id,
        status="running",
        message="Evaluation started.",
    )


@app.get("/api/evaluate/{run_id}", response_model=EvaluationJobResponse)
def evaluation_status(run_id: str) -> EvaluationJobResponse:
    status = read_run_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found.")
    result = status.get("result")
    return EvaluationJobResponse(
        run_id=run_id,
        status=status.get("status", "unknown"),
        message=status.get("message", ""),
        result=EvaluationResponse.model_validate(result) if result else None,
    )
