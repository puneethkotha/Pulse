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
        <div className="panel-header">
          <h2>Evaluation Metrics</h2>
        </div>
        <div className="panel-body">
          <p className="muted">Metrics not available. Run evaluate.py first.</p>
        </div>
      </section>
    );
  }

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
    BM25: "#475569",
    "Two-Tower": "#3b82f6",
    "Two-Tower + Re-ranker": "#10b981",
  };

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Evaluation Metrics</h2>
        <p style={{ marginTop: 6 }}>
          Computed on the held-out MovieLens 1M test set. Items rated ≥ 4.0 treated as relevant.
        </p>
      </div>
      <div className="panel-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2e42" vertical={false} />
            <XAxis dataKey="metric" tick={{ fill: "#7c92aa", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#7c92aa", fontSize: 11 }} axisLine={false} tickLine={false} width={36} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #2d3f55", color: "#f1f5f9", borderRadius: 6, fontSize: 12 }}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Legend wrapperStyle={{ color: "#7c92aa", fontSize: 12, paddingTop: 12 }} />
            {Object.entries(MODEL_LABELS).map(([, label]) => (
              <Bar key={label} dataKey={label} fill={modelColors[label]} radius={[2, 2, 0, 0]} maxBarSize={24} />
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
      </div>
    </section>
  );
}
