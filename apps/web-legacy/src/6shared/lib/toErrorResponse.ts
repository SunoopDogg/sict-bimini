import type { APIResponse } from '@/src/6shared/api/types';

export function toErrorResponse<T>(error: unknown, fallback: string): APIResponse<T> {
  return {
    success: false,
    data: null,
    error: error instanceof Error ? error.message : fallback,
  };
}
