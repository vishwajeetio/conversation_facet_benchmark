# Prompt Logs
These are the logs of the prompts used to build the conversation facet benchmark to speed up the development process and ensure consistency across implementations.

## Prompt 1: Minimal UI Implementation

```text
I already have a FastAPI backend design for a conversation facet benchmark.
Please write a minimal static UI using plain HTML, CSS, and JavaScript.

Requirements:
- One page only.
- Left panel: conversation JSON textarea, facet limit input, conversation id input.
- Right panel: score summary, facet count, status message, score table.
- One primary Evaluate button.
- Call GET /api/facets to load facet count and facet ids.
- Call POST /api/evaluate with conversation_id, turns, and selected facet_ids.
- Render turn_index, facet id/name, score, confidence, and rationale.
- Keep the UI modern, restrained, and easy to scan.
```

### Response Log

Implemented a focused static UI with a two-panel layout, sample conversation JSON, facet limit controls, evaluation call wiring, and a score table rendering score, confidence, and rationale.

### Files Changed

- Added `app/static/index.html` lines 1-62 for the UI structure.
- Added `app/static/app.js` lines 1-94 for facet loading, evaluation calls, and table rendering.
- Added `app/static/styles.css` lines 1-207 for responsive minimalist styling.

## Prompt 2: Facet Categorization Script

```text
Now please write a small Python script that categorizes facets through Ollama that fulfills the categorization requirements below:
- Read facets through the existing app facet loader.
- Use OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS from env.
- Use FACET_PROCESS_BATCH_SIZE for batching.
- Use FACET_MAX_CATEGORIES to cap the taxonomy.
- First ask the model for broad reusable category labels.
- Then assign every facet to only one of the allowed category labels.
- Write the updated CSV back to data/processed/facets.csv.
```

### Response Log

Added an Ollama-backed facet processing script that builds a capped taxonomy, assigns each facet to one allowed category, and writes the updated `data/processed/facets.csv`.

### Files Changed

- Added `app/facet_processing.py` lines 16-181 for taxonomy generation, capped category normalization, Ollama calls, and CSV writing.
- Added `scripts/process_facets.py` lines 1-32 for the Docker-runnable command.
- Added `docker-compose.yml` lines 10-11 for `FACET_PROCESS_BATCH_SIZE` and `FACET_MAX_CATEGORIES`.

## Prompt 3: Ollama Evaluation Client

```text
The benchmark should score one conversation turn against a batch of facets per model call. Please implement the client code for Ollama evaluation that fulfills the following requirements:
- Use urllib from the standard library.
- Score scale is exactly -2, -1, 0, 1, 2.
- Prefer 0 when the turn has no evidence for a facet.
- Include timeout handling and clear error messages.
```

### Response Log

Implemented the Ollama evaluator client with batched per-turn scoring, strict JSON output expectations, score/confidence normalization, timeout handling, and the five-point ordered score scale.

### Files Changed

- Added `app/evaluator.py` lines 15-214 for the score scale, Ollama client, prompt construction, JSON parsing, normalization, and batch loop.
- Added `app/main.py` lines 48-78 to expose `/api/evaluate` and save each run after scoring.
