# PinRanker

An end-to-end machine learning ranking system built on the MovieLens 1M dataset. The pipeline covers two-tower neural retrieval, approximate nearest-neighbor indexing with FAISS, pointwise re-ranking, a streaming feature layer over Kafka and Redis, a FastAPI inference service, and a local React dashboard.

---

## What it does

Given a user ID, PinRanker returns a ranked list of movie recommendations by:

1. Encoding the user into a 64-dimensional embedding with a trained two-tower model
2. Retrieving the top-100 candidate items by approximate nearest-neighbor search against the FAISS item index
3. Re-ranking the candidates with a learned pointwise MLP that incorporates item metadata, embedding similarity, and rolling user behavior features
4. Returning the top-K results with scores and item metadata

When Kafka and Redis are running, the system also accepts live interaction events and updates user features in real time without restarting the server.

---

## Dataset

**MovieLens 1M** — collected by GroupLens Research
Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip

| Property | Value |
|---|---|
| Users | 6,040 |
| Items | 3,706 |
| Total interactions | 1,000,209 |
| Ratings scale | 1–5 |

**Split strategy:** user-aware temporal split. For each user, interactions are sorted by timestamp and divided 80/10/10 into train, validation, and test sets. Users with fewer than 5 interactions are excluded.

---

## Architecture

```
MovieLens 1M
    │
    ▼
Preprocessing
  User features: gender, age bucket, occupation, zip prefix (30-dim)
  Item features: genre (18-dim), year, popularity, avg rating (21-dim)
    │
    ▼
Two-Tower Model (PyTorch)
  User tower: embedding + side features → 64-dim L2-normalized output
  Item tower: embedding + side features → 64-dim L2-normalized output
  Loss: in-batch negatives with temperature-scaled dot product
    │
    ▼
FAISS IVF Index
  Indexes all 3,706 item embeddings
  ANN search returns top-100 candidates per user query
    │
    ▼
Re-Ranker (Pointwise MLP)
  Input: 44-dim feature vector per (user, candidate item) pair
  Features: embedding similarity, genre overlap, popularity, item metadata,
            user history, rolling genre preferences
  Output: relevance score in [0, 1]
    │
    ▼
FastAPI Service
  GET /health
  GET /recommend?user_id=X&k=10
  POST /event
  GET /metrics
  GET /embedding?item_id=X
    │
    ▼
React + Vite Dashboard
  Overview · Demo · Metrics · Latency · Architecture
  Offline demo mode when backend is not running
```

Parallel streaming layer (optional):

```
Kafka Producer → user-interactions topic → Kafka Consumer → Redis Feature Store
```

---

## Evaluation results

Metrics computed on the held-out test set. Items rated ≥ 4.0 are treated as relevant.

| Metric | BM25 | Two-Tower | Two-Tower + Re-ranker |
|---|---|---|---|
| Precision@5 | 0.0101 | 0.0138 | 0.0284 |
| Precision@10 | 0.0100 | 0.0128 | 0.0250 |
| Recall@5 | 0.0071 | 0.0147 | 0.0272 |
| Recall@10 | 0.0160 | 0.0268 | 0.0471 |
| NDCG@5 | 0.0104 | 0.0169 | 0.0350 |
| NDCG@10 | 0.0133 | 0.0211 | 0.0408 |
| NDCG@20 | 0.0183 | 0.0297 | 0.0529 |

All numbers above are computed by `scripts/evaluate.py` and saved in `artifacts/metrics/evaluation_results.json`. The re-ranker improves NDCG@10 by approximately 2× over the two-tower baseline, which in turn outperforms BM25.

These are the actual numbers produced by this pipeline. They are not targets.

---

## Latency

Measured by `scripts/measure_latency.py` over 200 sequential requests to the `/recommend` endpoint (k=10, re-ranker enabled, Redis offline) on localhost.

| Percentile | Latency |
|---|---|
| Median | 6.04 ms |
| P90 | 7.31 ms |
| P95 | 8.66 ms |
| P99 | 15.78 ms |
| Mean | 6.38 ms |

Results saved in `artifacts/metrics/latency_report.json`.

---

## Setup

**Requirements:** Python 3.10+, Node 18+

```bash
# 1. Create a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download MovieLens 1M
bash scripts/download_data.sh

# 4. Preprocess the data
python scripts/preprocess.py

# 5. Train the two-tower model
python scripts/train_two_tower.py

# 6. Build the FAISS index
python scripts/build_index.py

# 7. Train the re-ranker
python scripts/train_reranker.py

# 8. Run evaluation
python scripts/evaluate.py

# 9. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 10. Measure latency (API must be running)
python scripts/measure_latency.py

# 11. Generate offline demo data
python scripts/generate_offline_demo.py
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in a browser. The dashboard works in offline demo mode (using precomputed data from `frontend/public/offline_data/`) when the backend is not running.

**Optional — streaming layer (requires Docker):**

```bash
docker compose up -d   # starts Kafka + Redis
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and component availability |
| GET | `/recommend?user_id=X&k=10` | Top-K recommendations for a user |
| POST | `/event` | Submit an interaction event |
| GET | `/metrics` | Evaluation results |
| GET | `/embedding?item_id=X` | Two-tower item embedding |

Example:

```bash
curl http://localhost:8000/recommend?user_id=1&k=5
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

Tests cover preprocessing, FAISS retrieval, ranking metrics, streaming event schema and feature updates, and all API endpoints.

---

## Limitations

- The two-tower model was trained for a limited number of epochs on CPU. NDCG scores are modest. Longer training, better hyperparameter tuning, or a GPU would improve them.
- The item catalog is small (3,706 items). FAISS IVF is overkill at this scale; for larger catalogs the indexing benefits become significant.
- BM25 query quality depends heavily on how well user genre preferences are captured from historical ratings. Users with thin history produce weak queries.
- The re-ranker uses offline user features by default. It will use Redis features when available, but the streaming layer requires Docker.
- Latency numbers reflect local single-process execution. They are not representative of a production deployment.
- All evaluation is done offline on a static test set. Online A/B testing is not implemented.

---

## Project structure

```
├── configs/          Model and training configuration
├── data/
│   ├── raw/          Downloaded MovieLens 1M files
│   ├── processed/    Preprocessed splits, feature matrices, ID maps
│   └── embeddings/   Saved item embeddings from two-tower model
├── models/
│   ├── two_tower/    Trained two-tower checkpoint
│   └── reranker/     Trained re-ranker checkpoint
├── indexes/faiss/    FAISS index and item ID map
├── artifacts/
│   ├── metrics/      Evaluation results and latency report (JSON + Markdown)
│   └── offline_demo/ Pre-computed data served to the frontend
├── src/
│   ├── data/         Loader, preprocessor, schema
│   ├── models/       Two-tower, re-ranker, BM25, feature builder
│   ├── indexing/     FAISS index builder and retriever
│   ├── evaluation/   Metrics and evaluator
│   ├── streaming/    Kafka producer, consumer, event schema
│   ├── features/     Redis online feature store
│   └── api/          FastAPI app and routers
├── scripts/          Training, evaluation, and utility scripts
├── tests/            Unit and integration tests
├── frontend/         React + TypeScript + Vite dashboard
└── docker-compose.yml Kafka + Redis local setup
```
