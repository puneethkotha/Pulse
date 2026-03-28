import { useState } from "react";
import type { RecommendResponse, SampleRecommendation } from "../types";
import { useRecommend } from "../hooks/useBackend";

interface RecommendPanelProps {
  backendUp: boolean;
  sampleRecs?: SampleRecommendation[];
}

export function RecommendPanel({ backendUp, sampleRecs }: RecommendPanelProps) {
  const [userId, setUserId] = useState("");
  const [k, setK] = useState(10);
  const { result, loading, error, recommend } = useRecommend();

  const [offlineUserId, setOfflineUserId] = useState<number | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const id = parseInt(userId, 10);
    if (!isNaN(id)) recommend(id, k);
  };

  const activeResult: RecommendResponse | null = result;

  const offlineResult =
    !backendUp && sampleRecs && offlineUserId !== null
      ? sampleRecs.find((r) => r.user_id === offlineUserId)
      : null;

  // When the live call fails, fall back silently to sample recs for that user
  const fallbackRec = error && sampleRecs
    ? sampleRecs.find((r) => r.user_id === parseInt(userId, 10))
    : null;

  const showRecs = backendUp
    ? activeResult?.recommendations ?? fallbackRec?.recommendations ?? []
    : offlineResult?.recommendations ?? [];

  const showingFallback = backendUp && !activeResult && !!fallbackRec;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Recommendation Demo</h2>
        <p style={{ marginTop: 6 }}>
          {backendUp
            ? "Enter a user ID to retrieve personalized recommendations from the live pipeline."
            : "Backend is offline — showing pre-computed recommendations from the pipeline."}
        </p>
      </div>

      <div className="panel-body">
        {backendUp ? (
          <div>
            <form onSubmit={handleSubmit} className="rec-form">
              <label>
                User ID
                <input
                  type="number"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="1 – 6040"
                  min={1}
                  max={6040}
                />
              </label>
              <label>
                Top K
                <input
                  type="number"
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value) || 10)}
                  min={1}
                  max={100}
                />
              </label>
              <button type="submit" disabled={loading}>
                {loading ? "Loading…" : "Get Recommendations"}
              </button>
            </form>
            {sampleRecs && sampleRecs.length > 0 && (
              <div className="quick-pick">
                <span className="quick-pick-label">Try a sample user:</span>
                <div className="user-chips">
                  {sampleRecs.map((r) => (
                    <button
                      key={r.user_id}
                      className={`chip ${userId === String(r.user_id) ? "chip-active" : ""}`}
                      onClick={() => setUserId(String(r.user_id))}
                      type="button"
                    >
                      User {r.user_id}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div>
            <div className="offline-note">
              Select a user below to view pre-computed recommendations.
            </div>
            {sampleRecs && (
              <div className="user-chips">
                {sampleRecs.map((r) => (
                  <button
                    key={r.user_id}
                    className={`chip ${offlineUserId === r.user_id ? "chip-active" : ""}`}
                    onClick={() => setOfflineUserId(r.user_id)}
                  >
                    User {r.user_id}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {showingFallback && (
          <div className="offline-note">
            Showing pre-computed recommendations for this user.
          </div>
        )}

        {error && !fallbackRec && <p className="error-msg">{error}</p>}

        {backendUp && activeResult && (
          <div className="rec-meta">
            <span>Latency <strong>{activeResult.latency_ms} ms</strong></span>
            <span>Source <strong>{activeResult.retrieval_source}</strong></span>
            <span>Online features <strong>{activeResult.online_features_used ? "yes" : "no"}</strong></span>
          </div>
        )}

        {showRecs.length > 0 && (
          <table className="rec-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Title</th>
                <th>Genres</th>
                <th>Avg Rating</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {showRecs.map((rec, i) => (
                <tr key={rec.item_id}>
                  <td>{i + 1}</td>
                  <td>{rec.title}</td>
                  <td>{rec.genres.join(", ")}</td>
                  <td>{rec.avg_rating.toFixed(2)}</td>
                  <td>{rec.score.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
