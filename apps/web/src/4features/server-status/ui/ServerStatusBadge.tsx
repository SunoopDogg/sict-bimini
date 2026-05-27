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
}

const POLL_INTERVAL = 30_000;

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
        const data = response.data;
        const state: ServerState =
          data.status === 'healthy' &&
          data.ollama_connected &&
          data.milvus_connected
            ? 'healthy'
            : 'degraded';
        setStatusInfo({ health: data, state });
      } else {
        setStatusInfo({ health: null, state: 'offline' });
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
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">{t.server.version}</span>
                <span className="font-mono text-xs">
                  {statusInfo.health.version}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Ollama</span>
                <span
                  className={
                    statusInfo.health.ollama_connected
                      ? 'text-green-500'
                      : 'text-red-500'
                  }
                >
                  {statusInfo.health.ollama_connected ? t.server.connected : t.server.notConnected}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Milvus</span>
                <span
                  className={
                    statusInfo.health.milvus_connected
                      ? 'text-green-500'
                      : 'text-red-500'
                  }
                >
                  {statusInfo.health.milvus_connected ? t.server.connected : t.server.notConnected}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">{t.server.cannotConnect}</p>
          )}
        </div>
      )}
    </div>
  );
}
