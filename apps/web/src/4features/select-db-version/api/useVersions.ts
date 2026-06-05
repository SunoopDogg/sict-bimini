'use client';

import { useCallback, useEffect, useState } from 'react';

import type { DbVersion } from '@/5entities/db-version';
import { fetchVersions } from '@/6shared/api';

export function useVersions() {
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setIsLoading(true);
    const res = await fetchVersions();
    if (res.success && res.data) {
      setVersions(res.data.versions);
      setError(null);
    } else {
      setError(res.error ?? '버전 목록 조회 실패');
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { versions, isLoading, error, refetch };
}
