---
name: PinRanker project state
description: What has been built and verified in the PinRanker ML ranking pipeline project
type: project
---

PinRanker is a complete end-to-end ML ranking system at `/Users/puneeth/Documents/Projects/Pulse`.

**Status: All phases complete and verified.**

## What's built and working

- MovieLens 1M downloaded and preprocessed (6040 users, 3706 items, 1M interactions)
- User-aware temporal 80/10/10 split with no leakage
- Two-tower PyTorch model trained (best val_loss=6.0745, 11 epochs with early stopping)
- FAISS IVF index built over 3706 item embeddings (64-dim)
- Pointwise re-ranker trained (44-dim features, best val_loss=0.0623)
- BM25 baseline using item title + genre text
- Full evaluation on test set — results saved to `artifacts/metrics/evaluation_results.json`
- FastAPI service with /health, /recommend, /event, /metrics, /embedding
- Latency measured: median 6.04ms, P99 15.78ms (200 requests, localhost)
- React + Vite + TypeScript dashboard with offline demo mode
- 60/60 tests passing

## Verified metrics (from evaluation_results.json)

| Model | NDCG@10 |
|---|---|
| BM25 | 0.0133 |
| Two-Tower | 0.0211 |
| Two-Tower + Re-ranker | 0.0408 |

## Key technical decisions

- Re-ranker uses 44-dim features (RERANKER_INPUT_DIM constant in reranker.py) — model_config.yaml input_dim was wrong (32 vs 44), fixed in dependencies.py to use the constant directly
- FAISS + PyTorch require KMP_DUPLICATE_LIB_OK=TRUE on this Mac (OpenMP conflict)
- API tests mock app_state.load to prevent FAISS import during pytest

**Why:** resume-worthy end-to-end ML ranking project, local only, no GitHub.
**How to apply:** All metrics and claims are backed by artifacts in artifacts/metrics/.
