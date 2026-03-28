export interface Recommendation {
  item_id: number;
  score: number;
  retrieval_score: number;
  title: string;
  genres: string[];
  avg_rating: number;
}

export interface RecommendResponse {
  user_id: number;
  recommendations: Recommendation[];
  latency_ms: number;
  retrieval_source: string;
  online_features_used: boolean;
}

export interface MetricsResult {
  [key: string]: number | string;
}

export interface EvalMetrics {
  bm25?: MetricsResult;
  two_tower?: MetricsResult;
  two_tower_reranker?: MetricsResult;
  [key: string]: MetricsResult | undefined;
}

export interface LatencyReport {
  num_requests: number;
  num_errors: number;
  mean_ms: number;
  median_ms: number;
  p90_ms: number;
  p95_ms: number;
  p99_ms: number;
  min_ms: number;
  max_ms: number;
}

export interface DatasetStats {
  num_users: number;
  num_items: number;
  num_train: number;
  num_val: number;
  num_test: number;
  total_interactions: number;
  item_feature_dim: number;
  user_feature_dim: number;
}

export interface ItemCatalogEntry {
  item_id: number;
  title: string;
  genres: string[];
  year: number;
  avg_rating: number;
  num_ratings: number;
}

export interface SampleRecommendation {
  user_id: number;
  recommendations: Recommendation[];
}

export interface OfflineData {
  metrics?: { results: EvalMetrics };
  latency?: LatencyReport;
  dataset_stats?: DatasetStats;
  item_catalog?: ItemCatalogEntry[];
  sample_recommendations?: SampleRecommendation[];
}

export interface HealthStatus {
  status: string;
  faiss_loaded: boolean;
  redis_available: boolean;
  reranker_loaded: boolean;
}
