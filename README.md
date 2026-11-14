# Token Optimization Platform

An enterprise LLM token optimization platform: it cuts the token (and dollar)
cost of LLM applications without silently degrading answer quality. Every
optimization decision is measurable — tokens saved, cost saved, quality
retained — and every policy change is versioned, evaluated on a golden set,
and gated before it can reach production traffic.

## The enterprise problem

LLM spend scales with tokens, and most production prompts are wasteful:
retrieved context repeats the same passage from two sources, prompts carry
filler and boilerplate, and every tenant shares one uncapped budget. The
naive fix — truncate the prompt — silently drops the facts the model needs.
This platform treats token spend as a governed resource:

- **Prompt compression** (extractive, redundancy-aware) shrinks prompts by
  deleting sentences, never rewriting them, so nothing hallucinated leaks in.
- **Semantic dedup** removes near-duplicate context blocks before they are
  paid for.
- **Budget-aware packing** fits context into a per-request token budget by
  value density, failing closed when mandatory context does not fit.
- **Caching-aware reuse**: a prefix cache accounts for shared prompt heads
  at a fraction of fresh cost; a semantic cache reuses answers for
  near-duplicate questions, skipping the model call entirely.
- **Cost / latency / quality evaluation** on a deterministic golden set,
  with keyword-coverage and ROUGE-L scorers — no model calls, fully
  reproducible.
- **Fail-closed budgets**: per-request and per-tenant (daily token and cost)
  ceilings raise instead of over-running.
- **Versioned policies with gated promotion**: a candidate policy becomes
  active only when its golden-set evaluation clears a quality floor and a
  savings floor.
- **Per-tenant metering** and a **hash-chained audit log** make every token
  and every decision accountable.

## Architecture overview

```
request ──▶ semantic cache ──▶ dedup context ──▶ pack by value density
              (hit = free)      (drop near-dups)    (fail-closed budget)
        ──▶ prefix cache accounting ──▶ extractive compression
        ──▶ budget enforcement (per-request + per-tenant)
        ──▶ metering ledger + hash-chained audit log
```

Pure-stdlib core (`src/tokenopt/`): `tokenizer` (deterministic token
counting/cost), `embeddings` (SHA-256 hashed embeddings, no model
downloads), `compression`, `dedup`, `packing`, `budgets`, `cache`,
`evaluation`, `policies`, `metering`, `audit`, `optimizer` (pipeline),
`config`, `demo_data` (synthetic golden set), `cli`, and `service`
(FastAPI, optional). See `docs/architecture.md` for the full design.

## How to run

```bash
make install        # create .venv and install dependencies
make test           # pytest (API tests auto-skip without FastAPI)
make demo           # end-to-end demo with measurable savings
make serve          # FastAPI on :8000
```

Or without a venv (core only, no FastAPI):

```bash
PYTHONPATH=src python3 scripts/demo.py
PYTHONPATH=src python3 -m pytest -q   # service tests skip
```

CLI:

```bash
PYTHONPATH=src python3 -m tokenopt.cli optimize --prompt "Summarize the refund policy..."
PYTHONPATH=src python3 -m tokenopt.cli evaluate
```

Docker:

```bash
docker build -t tokenopt .
docker run -p 8000:8000 tokenopt
```

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness + active policy |
| `POST /optimize` | Optimize a prompt + context (429 when over budget) |
| `POST /policies` | Register a versioned policy |
| `POST /policies/{name}/{v}/candidate` | Submit a draft as a candidate |
| `POST /policies/{name}/{v}/promote` | Gated promotion (422 when the gate fails) |
| `GET /policies/{name}/{v}/evaluate` | Golden-set evaluation |
| `GET /meter/{tenant}` | Per-tenant usage and savings |
| `GET /audit/verify` | Audit-chain integrity |

## Key design decisions and trade-offs

- **Extractive-only compression.** Sentences are deleted, never rewritten.
  This caps quality risk (no paraphrase drift, no invented facts) at the
  cost of lower maximum compression than abstractive methods.
- **Deterministic token counting.** A documented ~4-chars-per-token
  approximation instead of a real BPE tokenizer: no downloads, no drift,
  reproducible budgets and A/B comparisons. Absolute counts are estimates;
  relative savings are what the gates use.
- **Hashed embeddings, not a model.** Feature hashing with SHA-256 gives
  deterministic similarity for dedup and semantic caching with zero
  dependencies. It captures lexical overlap, not deep semantics — a known
  limit, documented in `docs/architecture.md`.
- **Greedy value-density packing, not optimal knapsack.** O(n log n),
  explainable, and good enough when scores already encode relevance.
  Required blocks are never dropped; an over-budget required block fails
  closed rather than silently degrading the prompt.
- **Modelled latency, not measured.** Evaluation latency is a documented
  per-token model so trade-offs are visible; it is not a benchmark claim.
- **Fail-closed everywhere.** Budget breaches, promotion-gate failures and
  over-compression (quality guard) all reject rather than degrade silently.
- **In-memory caches and ledger.** Simple, testable, and honest about the
  production step (Redis + a durable store) in the architecture doc.

## What it demonstrates

Token economics as an engineering discipline: measurable compression with
quality retention, budget enforcement as a safety property, policy
governance (version → evaluate → A/B → gated promote), per-tenant
metering, and tamper-evident auditing — the controls an enterprise needs
before it lets an optimizer touch production prompts.

## CI

The GitHub Actions workflow ships as `docs/github-actions-ci.yml`. To
activate it, move it to `.github/workflows/ci.yml` (the push integration
cannot write under `.github/workflows/` directly).
