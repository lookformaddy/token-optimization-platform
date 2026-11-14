# Architecture — Token Optimization Platform

## Goals

1. Cut LLM token spend measurably, without silently degrading quality.
2. Make every optimization decision auditable and every policy change gated.
3. Run fully offline with zero external dependencies in the core.

## Non-goals

- No model calls, no API keys, no network. The optimizer shapes prompts;
  it does not generate text.
- No real BPE tokenizer and no embedding model in v1 (see trade-offs).
- No persistent storage in v1: caches, ledger and audit log are in-memory
  and reset with the process.

## Components

```
src/tokenopt/
├── tokenizer.py    Deterministic token counting + cost math
├── embeddings.py   SHA-256 hashed embeddings + cosine similarity
├── compression.py  Extractive sentence compression + redundancy removal
├── dedup.py        Semantic dedup of context blocks
├── packing.py      Value-density knapsack packing under a token budget
├── budgets.py      Per-request / per-tenant ceilings, fail-closed
├── cache.py        Prefix cache + semantic cache, hit accounting
├── evaluation.py   Deterministic scorers (keyword coverage, ROUGE-L)
├── policies.py     Versioned policies, A/B, gated promotion
├── metering.py     Per-tenant usage ledger + aggregates
├── audit.py        Hash-chained append-only audit log
├── optimizer.py    Request pipeline (cache → dedup → pack → compress
│                   → enforce → meter → audit)
├── config.py       Environment-based settings
├── demo_data.py    Synthetic golden set + starter policies
├── cli.py          `optimize` / `evaluate` commands
└── service.py      FastAPI layer (optional dependency)
```

## Request pipeline

```
OptimizeRequest
  1. Semantic cache lookup — near-duplicate prompt reuses a stored answer;
     the model call is skipped entirely and the full prompt cost is booked
     as saved.
  2. Dedup — context blocks are clustered by hashed-embedding similarity;
     the first (most relevant) of each cluster is kept.
  3. Pack — required blocks first, then optional blocks by score/tokens,
     until the per-request budget is exhausted. A required block that does
     not fit raises BudgetExceeded (fail-closed).
  4. Prefix cache — the shared prompt head is accounted at the cached
     cost ratio (default 10% of fresh cost).
  5. Compress — sentences are scored (query overlap, content density,
     position) and the top-scoring set is kept to the policy's target
     ratio; near-duplicates among the kept set are dropped. A quality
     guard refuses compression that would keep too little content.
  6. Enforce — per-request and per-tenant daily token/cost ceilings are
     checked; any breach raises BudgetExceeded.
  7. Meter + audit — one ledger entry and one hash-chained audit entry.
```

## Token counting

`estimate_tokens` splits text into word/punctuation tokens; a word token
contributes `max(1, ceil(len / 4))` sub-tokens. This tracks the
~4-characters-per-token behaviour of common BPE tokenizers closely enough
for budgeting and relative savings, and it is deterministic: the same text
always yields the same count, so budgets and A/B comparisons are
reproducible. Absolute counts are estimates; gates use ratios, not absolutes.

## Semantic similarity

Hashed embeddings: each word is hashed with SHA-256 into a 128-dim vector
(feature hashing), L2-normalized; similarity is cosine. Deterministic,
dependency-free, and good at lexical overlap. It does **not** capture deep
semantics (synonyms, paraphrase beyond shared words) — a documented limit.
A production deployment can swap in a real embedding model behind the same
`embed`/`similarity` interface without touching callers.

## Policy governance

```
draft ──▶ candidate ──▶ promoted (active) ──▶ retired
                       ▲
                  gated on golden-set evaluation:
                  mean quality ≥ min_quality AND
                  mean savings ≥ min_savings AND
                  worst-case quality ≥ min_quality − 0.10
```

Promotion retires the previously active policy, so exactly one policy is
active. A/B comparison declares B the winner only when it saves more
without giving up more than 2 points of quality.

## Evaluation

- **Keyword coverage**: fraction of the golden answer's content words that
  survive in the optimized prompt — a proxy for "the model still has the
  facts it needs".
- **ROUGE-L F1**: longest-common-subsequence overlap between the optimized
  and original prompt; penalizes dropped ordering and content.
- **Latency**: modelled as `0.4 ms × input tokens + 8 ms` optimizer
  overhead. This makes the trade-off visible; it is not a measured
  benchmark.

## Budgets and failure modes

| Failure | Behaviour |
| --- | --- |
| Required block over budget | `BudgetExceeded`, request rejected |
| Per-request / daily token or cost breach | `BudgetExceeded`, request rejected |
| Compression would keep too little | Original prompt returned, `rejected=True` |
| Promotion gate missed | `PromotionBlocked`, policy stays a candidate |
| Audit chain tampered | `verify()` reports the first bad sequence |

Nothing fails open: there is no unbudgeted fallback path.

## Audit log

Append-only JSON lines. Each entry carries `prev_hash` and `entry_hash`
(SHA-256 over the canonical payload plus the previous hash). `verify()`
recomputes the chain and reports the first broken sequence. In production
the log should be written to append-only storage (object store / WORM
bucket); the chaining logic is unchanged.

## Production evolution

- Replace in-memory caches with Redis (prefix cache) and a vector store
  (semantic cache); keep the same accounting fields.
- Persist the metering ledger and audit log to a durable store; expose
  aggregates to the billing pipeline.
- Swap hashed embeddings for a hosted embedding model behind the same
  interface; re-run the golden set to re-tune thresholds.
- Add a real tokenizer (e.g. tiktoken) behind `estimate_tokens` for exact
  counts per model; keep the deterministic approximation as a fallback.
- Per-model pricing tables and per-tenant price overrides in `Settings`.
