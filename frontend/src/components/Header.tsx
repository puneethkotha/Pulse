interface HeaderProps {
  backendUp: boolean;
  offlineMode: boolean;
}

export function Header({ backendUp, offlineMode }: HeaderProps) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-title">
          <h1>PinRanker</h1>
          <span className="header-subtitle">Real-Time Embedding-Based Ranking Pipeline</span>
        </div>
        <div className="status-badges">
          <span className={`badge ${backendUp ? "badge-green" : "badge-red"}`}>
            {backendUp ? "Backend Online" : "Backend Offline"}
          </span>
          {offlineMode && (
            <span className="badge badge-yellow">Offline Demo Mode</span>
          )}
        </div>
      </div>
    </header>
  );
}
