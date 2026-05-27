# Ocean Across Conversation Facet Benchmark

Lightweight Dockerized benchmark for scoring every conversation turn across hundreds or thousands of evaluation facets using an open-weights model served by Ollama.

The system has simple FastAPI backend, plain HTML/CSS/JS UI, Ollama model service, CSV-backed facet registry, and JSON result files saved under `data/results`.

## Installation And Setup

### 1. Start The Services

```bash
docker compose up --build
```

This starts two services:

| Service | Purpose |
| --- | --- |
| `app` | FastAPI backend and minimal web UI. |
| `ollama` | Local model server for open-weights inference. |

The app is available at:

```text
http://localhost:8000
```

### 2. Pull The Model

In another terminal:

```bash
docker compose exec ollama ollama pull qwen2.5:7b-instruct
```

The default model is `qwen2.5:7b-instruct`, which keeps the assignment within the open-weights and <=16B constraint. You can switch models with `OLLAMA_MODEL` in `docker-compose.yml`.

### 3. Process The Facets

Run facet categorization once after the model is available:

```bash
docker compose run --rm app python scripts/process_facets.py
```

This reads `data/processed/facets.csv`, asks Ollama to assign compact category labels, and writes the updated CSV back to the host machine through the Docker volume:

```yaml
./data:/app/data
```

The app automatically creates `data/processed/facets.csv` from `data/raw/facets_assignment.csv` if the processed file is missing.

### 4. Limit Category Count

By default, facet processing uses at most 12 categories:

```yaml
FACET_MAX_CATEGORIES: 12
```

To override it for one run:

```bash
docker compose run --rm -e FACET_MAX_CATEGORIES=8 app python scripts/process_facets.py
```

## How To Use The Application

### UI Flow

Open:

```text
http://localhost:8000
```

Then:

1. Paste a conversation as JSON in the left panel.
2. Set `Facet limit` for a quick run, or increase it to score more facets.
3. Set a `Conversation id`.
4. Click `Evaluate`.
5. Review per-turn, per-facet scores in the table.

Expected conversation format:

```json
[
  {
    "speaker": "user",
    "text": "I checked the logs and can explain the tradeoffs clearly."
  },
  {
    "speaker": "assistant",
    "text": "Let's verify the risky step before continuing."
  }
]
```

### API Flow

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "demo",
    "turns": [
      {"speaker": "user", "text": "I checked the logs and can explain the tradeoffs clearly."}
    ],
    "facet_ids": ["F0001", "F0002", "F0003"]
  }'
```

Omit `facet_ids` to score all facets.

### Saved Results

Every evaluation run is saved automatically:

```text
data/results/<timestamp>_<conversation_id>.json
```

These files contain:

- Run id
- Conversation id
- Model name
- Input turns
- All facet scores
- Confidence values
- Rationales

Zip `data/results` manually when preparing the final submission artifact.

## Features

- Docker Compose setup with app and Ollama services.
- Open-weights local model inference through Ollama.
- Minimal UI for manual evaluation.
- API endpoint for programmatic evaluation.
- CSV-backed facet registry.
- Ollama-assisted facet categorization.
- Configurable maximum category count.
- Five ordered integer score scale: `-2, -1, 0, 1, 2`.
- Confidence output for every score.
- Short rationale for every score.
- Automatic result persistence as JSON.
- Batch-based scoring, so the architecture does not depend on one huge prompt.

## How It Works

### Data Flow

```text
raw facet CSV
  -> processed facet registry
  -> optional Ollama category labels
  -> conversation turns
  -> turn-by-turn facet batches
  -> Ollama JSON scoring
  -> normalized result records
  -> saved JSON run file
```

### Facet Registry

`data/processed/facets.csv` has only the columns needed at runtime:

| Column | Purpose |
| --- | --- |
| `facet_id` | Stable id such as `F0001`. |
| `name` | Cleaned facet name. |
| `category` | Broad Ollama-generated category hint. |

Categories are hints, not hard-coded logic. If the facet list changes, rerun `scripts/process_facets.py`.

### Evaluation Loop

For each request, the app:

1. Loads facets from `data/processed/facets.csv`.
2. Selects requested facets or all facets.
3. Iterates through each conversation turn.
4. Splits facets into batches using `FACET_BATCH_SIZE`.
5. Sends each turn-plus-facet-batch to Ollama.
6. Parses JSON response.
7. Normalizes missing or invalid scores.
8. Saves the run to `data/results`.

This satisfies the no one-shot prompt constraint because each model call scores only a manageable batch of facets for one turn.

## Scaling To 5000 Facets

The architecture scales by increasing the number of batches, not by changing the design.

The main work formula is:

```text
number_of_turns × ceil(number_of_facets / FACET_BATCH_SIZE)
```

Example with 5000 facets, 10 turns, and `FACET_BATCH_SIZE=8`:

```text
10 × ceil(5000 / 8) = 6250 Ollama calls
```

That is operationally heavy, but it does not require an architectural rewrite. The same code path works for 300 facets or 5000 facets.

## Parameter Tuning

| Variable | Default | Use |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used for categorization and scoring. |
| `FACET_BATCH_SIZE` | `8` | Number of facets scored per model call. |
| `FACET_PROCESS_BATCH_SIZE` | `30` | Number of facets categorized per model call. |
| `FACET_MAX_CATEGORIES` | `12` | Maximum category labels generated during facet processing. |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Timeout for one Ollama request. |
| `OLLAMA_NUM_CTX` | `4096` | Context window requested from Ollama. |

### Recommended Settings

For quick local demos:

```yaml
FACET_BATCH_SIZE: 8
FACET_MAX_CATEGORIES: 8
```

For better throughput on stronger hardware:

```yaml
FACET_BATCH_SIZE: 16
FACET_PROCESS_BATCH_SIZE: 50
OLLAMA_NUM_CTX: 8192
```

For safer JSON reliability on weaker hardware:

```yaml
FACET_BATCH_SIZE: 4
OLLAMA_TIMEOUT_SECONDS: 600
```

## What Limits Speed

The main bottleneck is local LLM generation. Every batch requires prompt processing and JSON generation.

Important limiting factors:

- Model size: larger models are slower.
- Hardware: CPU-only inference is much slower than GPU-backed Ollama.
- Facet batch size: larger batches reduce request count but increase prompt and output length.
- Conversation length: longer turns increase prompt tokens.
- Rationales: every rationale adds output tokens.
- Number of turns: each turn is scored independently.
- JSON reliability: very large batches increase malformed JSON risk.

## How To Improve Speed

Practical improvements:

- Increase `FACET_BATCH_SIZE` until JSON reliability starts dropping.
- Use GPU acceleration for Ollama.
- Use a smaller <=16B model if quality is acceptable.
- Reduce rationale length or make rationale optional for bulk scoring.
- Cache repeated `(turn, facet_id, model)` evaluations.
- Run independent batches in parallel with a worker queue.