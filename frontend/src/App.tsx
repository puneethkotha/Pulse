import { useState } from "react";
import { Header } from "./components/Header";
import { OverviewPanel } from "./components/OverviewPanel";
import { RecommendPanel } from "./components/RecommendPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { LatencyPanel } from "./components/LatencyPanel";
import { ArchitecturePanel } from "./components/ArchitecturePanel";
import { useHealth, useOfflineData, useMetrics } from "./hooks/useBackend";
import type { EvalMetrics } from "./types";
import "./App.css";

const NAV_ITEMS = ["Overview", "Demo", "Metrics", "Latency", "Architecture"];

function App() {
  const [activeTab, setActiveTab] = useState("Overview");
  const { backendUp } = useHealth();
  const offlineData = useOfflineData();
  const liveMetrics = useMetrics();

  const displayMetrics: EvalMetrics | undefined =
    liveMetrics ??
    (offlineData.metrics?.results as EvalMetrics | undefined) ??
    undefined;

  return (
    <div className="app">
      <Header backendUp={backendUp} offlineMode={!backendUp} />
      <nav className="nav">
        {NAV_ITEMS.map((tab) => (
          <button
            key={tab}
            className={`nav-btn ${activeTab === tab ? "nav-btn-active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>
      <main className="main">
        {activeTab === "Overview" && (
          <OverviewPanel stats={offlineData.dataset_stats} />
        )}
        {activeTab === "Demo" && (
          <RecommendPanel
            backendUp={backendUp}
            sampleRecs={offlineData.sample_recommendations}
          />
        )}
        {activeTab === "Metrics" && <MetricsPanel metrics={displayMetrics} />}
        {activeTab === "Latency" && (
          <LatencyPanel latency={offlineData.latency} />
        )}
        {activeTab === "Architecture" && <ArchitecturePanel />}
      </main>
    </div>
  );
}

export default App;
