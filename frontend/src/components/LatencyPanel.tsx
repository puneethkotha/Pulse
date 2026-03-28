import type { LatencyReport } from "../types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface LatencyPanelProps {
  latency?: LatencyReport | null;
}

export function LatencyPanel({ latency }: LatencyPanelProps) {
  if (!latency) {
    return (
      <section className="panel">
        <h2>Latency Summary</h2>
        <p className="muted">
          Latency data not available. Run scripts/measure_latency.py while the
          API is running.
        </p>
      </section>
    );
  }

  const chartData = [
    { percentile: "Min", ms: latency.min_ms },
    { percentile: "Median", ms: latency.median_ms },
    { percentile: "Mean", ms: latency.mean_ms },
    { percentile: "P90", ms: latency.p90_ms },
    { percentile: "P95", ms: latency.p95_ms },
    { percentile: "P99", ms: latency.p99_ms },
    { percentile: "Max", ms: latency.max_ms },
  ];

  return (
    <section className="panel">
      <h2>Latency Summary</h2>
      <p className="muted">
        End-to-end /recommend endpoint latency measured over {latency.num_requests} requests.
        {latency.num_errors > 0 && ` (${latency.num_errors} errors excluded)`}
      </p>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="percentile" tick={{ fill: "#94a3b8", fontSize: 12 }} />
          <YAxis
            unit="ms"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#94a3b8" }}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", color: "#f1f5f9" }}
            formatter={(v) => [`${v} ms`]}
          />
          <Bar dataKey="ms" fill="#3b82f6" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <div className="stats-grid" style={{ marginTop: "1rem" }}>
        <StatCard label="Mean" value={`${latency.mean_ms} ms`} />
        <StatCard label="Median" value={`${latency.median_ms} ms`} />
        <StatCard label="P90" value={`${latency.p90_ms} ms`} />
        <StatCard label="P95" value={`${latency.p95_ms} ms`} />
        <StatCard label="P99" value={`${latency.p99_ms} ms`} />
        <StatCard label="Requests" value={latency.num_requests.toLocaleString()} />
      </div>
    </section>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
