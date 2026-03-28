export function ArchitecturePanel() {
  const layers = [
    {
      name: "Data Layer",
      color: "#475569",
      items: [
        "MovieLens 1M (1M interactions, 6040 users, 3706 items)",
        "User-aware temporal split: 80/10/10",
        "Item features: genre (18-dim), year, popularity, avg rating",
        "User features: gender, age bucket, occupation, zip prefix",
      ],
    },
    {
      name: "Offline Training",
      color: "#1d4ed8",
      items: [
        "Two-tower PyTorch model — user tower + item tower",
        "In-batch negative contrastive loss (temperature=0.07)",
        "64-dim L2-normalized output embeddings",
        "Pointwise re-ranker MLP (44-dim input, BCELoss)",
        "BM25 baseline over title + genre text",
      ],
    },
    {
      name: "Indexing",
      color: "#7c3aed",
      items: [
        "FAISS IVF index over item embeddings",
        "ANN retrieval — top-100 candidates per query",
        "Re-ranker scores and sorts final top-K",
      ],
    },
    {
      name: "Streaming",
      color: "#0f766e",
      items: [
        "Kafka producer — simulates/replays interaction events",
        "Kafka consumer — updates rolling user features",
        "Redis feature store — genre counts, avg rating, last items",
        "Graceful fallback when Kafka/Redis unavailable",
      ],
    },
    {
      name: "Serving",
      color: "#b45309",
      items: [
        "FastAPI — GET /recommend, POST /event, GET /metrics",
        "Reads online features from Redis if available",
        "FAISS retrieval → re-ranker → ranked response",
        "Latency measured end-to-end per request",
      ],
    },
  ];

  return (
    <section className="panel">
      <h2>Architecture</h2>
      <div className="arch-layers">
        {layers.map((layer) => (
          <div key={layer.name} className="arch-layer" style={{ borderLeftColor: layer.color }}>
            <h4 style={{ color: layer.color }}>{layer.name}</h4>
            <ul>
              {layer.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
