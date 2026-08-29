# magicpin AI Challenge: Vera Bot Submission

Team Name: Team Vera Challenger  
Model: gemini-2.5-flash  
Approach: Stateful asynchronous composer with heuristic intent and auto-reply routing

## Architecture Overview

Our submission implements a stateful HTTP server built for the magicpin Judge Harness.

### 1. Dual Composition Pipelines
- Proactive Engine (`/v1/tick`): Combines CategoryContext, MerchantContext, and TriggerContext. Prompting prioritizes specificity and single low-friction CTAs. Asynchronous batching with `asyncio.gather()` processes multiple triggers concurrently to meet strict latency constraints.
- Reactive Engine (`/v1/reply`): Analyzes conversation history and merchant responses. Automatically switches from qualification to immediate action upon commitment.

### 2. State Management
- Ingestion (`/v1/context`): Keyed by `(scope, context_id)` pairs. Compares version numbers to prevent stale updates and ensure idempotency.
- History: In-memory store per `conversation_id`.

## Key Tradeoffs and Guardrails

1. Auto-Reply Heuristics: Pattern matching detects canned WhatsApp Business responses and loop repeats to exit gracefully without wasting token budget.
2. Direct Context Ingestion: Feeds structured context directly into the model to minimize latency.
3. Grounded Prompts: Constrains generation to verifiable context data, preventing fabricated citations or metrics. Adheres to category-specific vocabulary and code-mixing preferences.

## What Additional Context Would Help?

- Live inventory and booking APIs for instant appointment slot confirmation.
- Aggregated local peer performance benchmarks for sharper social proof copy.

## Running Locally

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```
