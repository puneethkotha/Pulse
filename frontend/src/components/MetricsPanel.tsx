import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { EvalMetrics } from "../types";

interface MetricsPanelProps {
  metrics?: EvalMetrics | null;
}

const MODEL_LABELS: Record<string, string> = {
  bm25: "BM25",
  two_tower: "Two-Tower",
  two_tower_reranker: "Two-Tower + Re-ranker",
};

const METRIC_KEYS = [
  "ndcg@5",
  "ndcg@10",
  "ndcg@20",
  "precision@5",
  "precision@10",
  "recall@5",
  "recall@10",
];

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  if (!metrics) {
    return (
      <section className="panel">
        <h2>Evaluation Metrics</h2>
        <p className="muted">Metrics not available. Run evaluate.py first.</p>
      </section>
    );
  }

  // Build chart data: one entry per metric key
  const chartData = METRIC_KEYS.map((metricKey) => {
    const entry: Record<string, string | number> = { metric: metricKey };
    for (const [modelKey, modelLabel] of Object.entries(MODEL_LABELS)) {
      const val = metrics[modelKey]?.[metricKey];
      if (typeof val === "number") {
        entry[modelLabel] = parseFloat(val.toFixed(4));
      }
    }
    return entry;
  });

  const modelColors: Record<string, string> = {
    BM25: "#94a3b8",
    "Two-Tower": "#3b82f6",
    "Two-Tower + Re-ranker": "#10b981",
  };

  return (
    <section className="panel">
      <h2>Evaluation Metrics</h2>
      <p className="muted">
        Computed on the held-out MovieLens 1M test set. Items rated ≥ 4.0 are
        treated as relevant.
      </p>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", color: "#f1f5f9" }}
          />
          <Legend wrapperStyle={{ color: "#94a3b8" }} />
          {Object.entries(MODEL_LABELS).map(([, label]) => (
            <Bar key={label} dataKey={label} fill={modelColors[label]} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <div className="metrics-table-wrap">
        <table className="rec-table">
          <thead>
            <tr>
              <th>Metric</th>
              {Object.values(MODEL_LABELS).map((l) => (
                <th key={l}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRIC_KEYS.map((key) => (
              <tr key={key}>
                <td>
                  <code>{key}</code>
                </td>
                {Object.keys(MODEL_LABELS).map((mk) => {
                  const val = metrics[mk]?.[key];
                  return (
                    <td key={mk}>
                      {typeof val === "number" ? val.toFixed(4) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
