import { useState, useEffect, useCallback } from "react";
import type {
  RecommendResponse,
  EvalMetrics,
  HealthStatus,
  OfflineData,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const OFFLINE_BASE = "/offline_data";

async function tryFetch<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export function useHealth() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [backendUp, setBackendUp] = useState(false);

  useEffect(() => {
    tryFetch<HealthStatus>(`${API_BASE}/health`).then((h) => {
      setHealth(h);
      setBackendUp(h?.status === "ok");
    });
  }, []);

  return { health, backendUp };
}

export function useOfflineData() {
  const [data, setData] = useState<OfflineData>({});

  useEffect(() => {
    const files = [
      "metrics.json",
      "latency.json",
      "dataset_stats.json",
      "item_catalog.json",
      "sample_recommendations.json",
    ];

    Promise.all(
      files.map((f) =>
        tryFetch<unknown>(`${OFFLINE_BASE}/${f}`).then((v) => [f, v] as const)
      )
    ).then((results) => {
      const d: OfflineData = {};
      for (const [fname, val] of results) {
        if (val === null) continue;
        if (fname === "metrics.json") d.metrics = val as OfflineData["metrics"];
        if (fname === "latency.json") d.latency = val as OfflineData["latency"];
        if (fname === "dataset_stats.json") d.dataset_stats = val as OfflineData["dataset_stats"];
        if (fname === "item_catalog.json") d.item_catalog = val as OfflineData["item_catalog"];
        if (fname === "sample_recommendations.json")
          d.sample_recommendations = val as OfflineData["sample_recommendations"];
      }
      setData(d);
    });
  }, []);

  return data;
}

export function useRecommend() {
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recommend = useCallback(async (userId: number, k: number = 10) => {
    setLoading(true);
    setError(null);
    const data = await tryFetch<RecommendResponse>(
      `${API_BASE}/recommend?user_id=${userId}&k=${k}`
    );
    if (data) {
      setResult(data);
    } else {
      setError("Backend unavailable or user not found.");
    }
    setLoading(false);
  }, []);

  return { result, loading, error, recommend };
}

export function useMetrics() {
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null);

  useEffect(() => {
    tryFetch<EvalMetrics>(`${API_BASE}/metrics`).then((m) => setMetrics(m));
  }, []);

  return metrics;
}
