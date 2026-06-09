'use client';

import { useEffect, useRef, useState } from 'react';

import type { HealthStatus } from '@/5entities/health';
import { checkHealth } from '@/6shared/api';
import { useLocale } from '@/6shared/i18n';
import { cn } from '@/6shared/lib/cn';
import { Badge } from '@/6shared/ui/primitive/badge';

type ServerState = 'healthy' | 'degraded' | 'offline';

interface StatusInfo {
  health: HealthStatus | null;
  state: ServerState;
  errorMessage?: string;
}

const POLL_INTERVAL = 30_000;

const SERVICE_LABELS: Record<string, string> = {
  qdrant: 'Vector DB',
  llm: 'LLM',
};

export function ServerStatusBadge() {
  const [statusInfo, setStatusInfo] = useState<StatusInfo>({
    health: null,
    state: 'offline',
  });
  const [showDetail, setShowDetail] = useState(false);
  const badgeRef = useRef<HTMLDivElement>(null);
  const { t } = useLocale();

  const stateConfig: Record<ServerState, { dotClass: string; label: string }> = {
    healthy: { dotClass: 'bg-green-500', label: t.server.online },
    degraded: { dotClass: 'bg-yellow-500', label: t.server.degraded },
    offline: { dotClass: 'bg-red-500', label: t.server.offline },
  };

  useEffect(() => {
    let active = true;

    const poll = async () => {
      const response = await checkHealth();
      if (!active) return;
      if (response.success && response.data) {
        const state: ServerState = response.data.status === 'ok' ? 'healthy' : 'degraded';
        setStatusInfo({ health: response.data, state });
      } else {
        setStatusInfo({ health: null, state: 'offline', errorMessage: response.error ?? undefined });
      }
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!showDetail) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (badgeRef.current && !badgeRef.current.contains(e.target as Node)) {
        setShowDetail(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showDetail]);

  const { dotClass, label } = stateConfig[statusInfo.state];

  return (
    <div className="relative" ref={badgeRef}>
      <Badge
        variant="outline"
        className="cursor-pointer gap-1.5 select-none"
        onClick={() => setShowDetail((prev) => !prev)}
      >
        <span className={cn('inline-block h-2 w-2 rounded-full', dotClass)} />
        {label}
      </Badge>

      {showDetail && (
        <div className="bg-popover text-popover-foreground absolute top-full left-0 z-50 mt-2 w-56 rounded-md border p-3 text-sm shadow-md">
          {statusInfo.health ? (
            <div className="space-y-2">
              {Object.entries(statusInfo.health.services).map(([name, svc]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="text-muted-foreground capitalize">
                    {SERVICE_LABELS[name] ?? name}
                  </span>
                  <span className={svc.status === 'ok' ? 'text-green-500' : 'text-red-500'}>
                    {svc.status === 'ok'
                      ? t.server.connected
                      : svc.detail ?? t.server.notConnected}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-xs">
              {statusInfo.errorMessage ?? t.server.cannotConnect}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
