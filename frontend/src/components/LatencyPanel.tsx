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
        <div className="panel-header">
          <h2>Latency</h2>
        </div>
        <div className="panel-body">
          <p className="muted">
            Latency data not available. Run scripts/measure_latency.py while the API is running.
          </p>
        </div>
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
      <div className="panel-header">
        <h2>Latency</h2>
        <p style={{ marginTop: 6 }}>
          End-to-end /recommend latency over {latency.num_requests} requests.
          {latency.num_errors > 0 && ` ${latency.num_errors} errors excluded.`}
        </p>
      </div>
      <div className="panel-body">
        <div className="stats-grid" style={{ marginBottom: 24 }}>
          <StatCard label="Mean" value={`${latency.mean_ms} ms`} />
          <StatCard label="Median" value={`${latency.median_ms} ms`} />
          <StatCard label="P90" value={`${latency.p90_ms} ms`} />
          <StatCard label="P95" value={`${latency.p95_ms} ms`} />
          <StatCard label="P99" value={`${latency.p99_ms} ms`} />
          <StatCard label="Requests" value={latency.num_requests.toLocaleString()} />
        </div>

        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2e42" vertical={false} />
            <XAxis dataKey="percentile" tick={{ fill: "#7c92aa", fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis
              unit="ms"
              tick={{ fill: "#7c92aa", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #2d3f55", color: "#f1f5f9", borderRadius: 6, fontSize: 12 }}
              formatter={(v) => [`${v} ms`]}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar dataKey="ms" fill="#3b82f6" radius={[2, 2, 0, 0]} maxBarSize={32} />
          </BarChart>
        </ResponsiveContainer>
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
