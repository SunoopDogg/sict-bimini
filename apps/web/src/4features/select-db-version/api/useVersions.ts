'use client';

import { useEffect, useState } from 'react';

import type { DbVersion } from '@/5entities/db-version';
import { fetchVersions } from '@/6shared/api';

export function useVersions() {
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchVersions().then((res) => {
      if (!active) return;
      if (res.success && res.data) {
        setVersions(res.data.versions);
      } else {
        setError(res.error ?? '버전 목록 조회 실패');
      }
      setIsLoading(false);
    });
    return () => {
      active = false;
    };
  }, []);

  return { versions, isLoading, error };
}
