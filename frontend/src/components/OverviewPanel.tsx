import type { DatasetStats } from "../types";

interface OverviewPanelProps {
  stats?: DatasetStats;
}

export function OverviewPanel({ stats }: OverviewPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Project Overview</h2>
        <p style={{ marginTop: 6 }}>
          An end-to-end ML ranking system trained on the MovieLens 1M dataset.
          The pipeline includes a two-tower neural retrieval model, a FAISS
          approximate nearest-neighbor index, a pointwise re-ranker, and a
          streaming layer that updates user features in real time via Kafka and
          Redis.
        </p>
      </div>

      <div className="panel-body">
        {stats && (
          <div className="stats-grid">
            <StatCard label="Users" value={stats.num_users.toLocaleString()} />
            <StatCard label="Items" value={stats.num_items.toLocaleString()} />
            <StatCard label="Interactions" value={stats.total_interactions.toLocaleString()} />
            <StatCard label="Train" value={stats.num_train.toLocaleString()} />
            <StatCard label="Val" value={stats.num_val.toLocaleString()} />
            <StatCard label="Test" value={stats.num_test.toLocaleString()} />
          </div>
        )}

        <h3>Pipeline</h3>
        <div className="arch-list">
          {[
            ["Data", "MovieLens 1M — user-aware temporal train/val/test split"],
            ["Retrieval", "Two-tower PyTorch model with in-batch negatives, FAISS IVF index"],
            ["Re-ranking", "Pointwise MLP re-ranker over 44 candidate features"],
            ["Baseline", "BM25 over item title + genre text"],
            ["Streaming", "Kafka producer/consumer updating rolling user features in Redis"],
            ["API", "FastAPI — /health, /recommend, /event, /metrics, /embedding"],
          ].map(([label, desc]) => (
            <div className="arch-row" key={label}>
              <span className="arch-label">{label}</span>
              <span className="arch-desc">{desc}</span>
            </div>
          ))}
        </div>
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
