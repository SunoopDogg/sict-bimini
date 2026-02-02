'use client';

import { useEffect, useRef, useState } from 'react';

import type { HealthStatus } from '@/src/5entities/health';
import { checkHealth } from '@/src/6shared/api';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import { cn } from '@/src/6shared/lib/cn';

type ServerState = 'healthy' | 'degraded' | 'offline';

interface StatusInfo {
  health: HealthStatus | null;
  state: ServerState;
}

const POLL_INTERVAL = 30_000;

const stateConfig: Record<ServerState, { dotClass: string; label: string }> = {
  healthy: { dotClass: 'bg-green-500', label: '서버 정상' },
  degraded: { dotClass: 'bg-yellow-500', label: '서버 불안정' },
  offline: { dotClass: 'bg-red-500', label: '서버 오프라인' },
};

export function ServerStatusBadge() {
  const [statusInfo, setStatusInfo] = useState<StatusInfo>({
    health: null,
    state: 'offline',
  });
  const [showDetail, setShowDetail] = useState(false);
  const badgeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      const response = await checkHealth();
      if (!active) return;
      if (response.success && response.data) {
        const data = response.data;
        const state: ServerState =
          data.status === 'healthy' && data.ollama_connected && data.milvus_connected
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
        <div className="absolute top-full left-0 z-50 mt-2 w-56 rounded-md border bg-popover p-3 text-sm text-popover-foreground shadow-md">
          {statusInfo.health ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">버전</span>
                <span className="font-mono text-xs">{statusInfo.health.version}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Ollama</span>
                <span className={statusInfo.health.ollama_connected ? 'text-green-500' : 'text-red-500'}>
                  {statusInfo.health.ollama_connected ? '연결됨' : '연결 안됨'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Milvus</span>
                <span className={statusInfo.health.milvus_connected ? 'text-green-500' : 'text-red-500'}>
                  {statusInfo.health.milvus_connected ? '연결됨' : '연결 안됨'}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">서버에 연결할 수 없습니다.</p>
          )}
        </div>
      )}
    </div>
  );
}
