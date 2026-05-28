export interface ServiceStatus {
  status: 'ok' | 'error';
  detail: string | null;
}

export interface HealthStatus {
  status: 'ok' | 'degraded';
  services: Record<string, ServiceStatus>;
}
