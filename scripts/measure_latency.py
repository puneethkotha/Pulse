"""
API latency measurement script.

Sends recommendation requests to the running FastAPI service and measures
end-to-end latency. Results are saved to artifacts/metrics/latency_report.json.

Usage:
    python scripts/measure_latency.py [--url http://localhost:8000] [--n 200]
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts/metrics")


def load_sample_user_ids(n: int = 100) -> list[int]:
    """Load real user IDs from processed data."""
    import pickle
    with open("data/processed/user_id_map.pkl", "rb") as f:
        user_id_map = pickle.load(f)
    all_ids = list(user_id_map.keys())
    random.shuffle(all_ids)
    return all_ids[:n]


def measure_latency(
    url: str,
    user_ids: list[int],
    k: int = 10,
    warmup: int = 10,
) -> dict:
    latencies = []

    # warmup
    for uid in user_ids[:warmup]:
        try:
            httpx.get(f"{url}/recommend", params={"user_id": uid, "k": k}, timeout=10)
        except Exception:
            pass

    # timed requests
    errors = 0
    for uid in user_ids:
        t0 = time.perf_counter()
        try:
            resp = httpx.get(f"{url}/recommend", params={"user_id": uid, "k": k}, timeout=10)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed_ms)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            logger.warning("Request failed for user %d: %s", uid, e)

    if not latencies:
        return {"error": "No successful requests"}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    report = {
        "num_requests": n,
        "num_errors": errors,
        "mean_ms": round(sum(latencies) / n, 2),
        "median_ms": round(latencies_sorted[n // 2], 2),
        "p90_ms": round(latencies_sorted[int(n * 0.90)], 2),
        "p95_ms": round(latencies_sorted[int(n * 0.95)], 2),
        "p99_ms": round(latencies_sorted[int(n * 0.99)], 2),
        "min_ms": round(latencies_sorted[0], 2),
        "max_ms": round(latencies_sorted[-1], 2),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="Number of requests")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    logger.info("Checking service health at %s/health ...", args.url)
    try:
        resp = httpx.get(f"{args.url}/health", timeout=5)
        logger.info("Health: %s", resp.json())
    except Exception as e:
        logger.error("Service not reachable: %s", e)
        return

    logger.info("Loading user IDs...")
    user_ids = load_sample_user_ids(n=args.n)

    logger.info("Measuring latency for %d requests (k=%d)...", len(user_ids), args.k)
    report = measure_latency(args.url, user_ids, k=args.k)

    logger.info("Results:")
    for k, v in report.items():
        logger.info("  %s: %s", k, v)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "latency_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Latency report saved to artifacts/metrics/latency_report.json")


if __name__ == "__main__":
    main()
