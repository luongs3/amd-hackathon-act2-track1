# Track 1 — Hybrid Token-Efficient Routing Agent
**AMD Developer Hackathon: ACT II** · Team: AuditAgent

## What it does
An AI agent that classifies each incoming task into one of 8 capability categories, then routes it to the **cheapest Fireworks AI model that can still answer accurately** — minimizing total token spend while maintaining high-quality responses.

## How it works
```
/input/tasks.json  →  [classify]  →  [route to cheapest model]  →  /output/results.json
```

**8 categories handled:** NER · SENTIMENT · SUMMARIZATION · CODE_DEBUG · CODE_GEN · LOGIC · MATH · FACTUAL

**Routing strategy:**
- Fast/cheap models → SENTIMENT, NER, FACTUAL (simple output shapes)
- Strong models → CODE, MATH, LOGIC, SUMMARIZATION (accuracy-critical)
- Keyword-based classifier with code-fence detection for robust categorization
- Automatic fallback chain if a model fails or is unavailable
- 429 rate-limit backoff, 4 retries per model, 9-min global deadline guard

## Running (Docker)
```bash
docker build -t amd-track1 .
docker run --rm \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=llama-v3p1-8b-instruct,llama-v3p1-70b-instruct \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  amd-track1
```

## Input / Output format
**Input** `/input/tasks.json`:
```json
[{"task_id": "t1", "prompt": "What is the capital of France?"}]
```

**Output** `/output/results.json`:
```json
[{"task_id": "t1", "response": "Paris.", "category": "FACTUAL", "model": "...", "usage": {...}}]
```

## Tech
- Python 3.12, `requests`
- Pure routing logic — no fine-tuning, no local models
- All inference via Fireworks AI API (FIREWORKS_BASE_URL)
