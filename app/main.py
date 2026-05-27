from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.evaluator import SCORE_SCALE, evaluate_conversation, make_client
from app.facets import load_facets, select_facets
from app.models import EvaluationRequest, EvaluationResponse
from app.storage import save_run


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


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    selected = select_facets(payload.facet_ids)
    if not selected:
        raise HTTPException(status_code=400, detail="No matching facets found.")
    client = make_client()
    try:
        results = await evaluate_conversation(
            payload.conversation_id,
            payload.turns,
            selected,
            client=client,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    run_id, saved_path = save_run(
        payload.conversation_id,
        client.model_name,
        payload.turns,
        results,
    )
    return EvaluationResponse(
        conversation_id=payload.conversation_id,
        run_id=run_id,
        score_scale=SCORE_SCALE,
        facet_count=len(selected),
        turn_count=len(payload.turns),
        model=client.model_name,
        saved_path=str(saved_path),
        results=results,
    )
