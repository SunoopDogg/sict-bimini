export interface HealthStatus {
  status: 'healthy' | 'degraded';
  version: string;
  ollama_connected: boolean;
  milvus_connected: boolean;
}
