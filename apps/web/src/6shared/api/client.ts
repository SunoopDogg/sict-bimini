import type { BIMAttributeListResponse } from '@/5entities/bim-attribute';
import type { BIMObject } from '@/5entities/bim-object';
import type { DbVersionListResponse } from '@/5entities/db-version';
import type { HealthStatus } from '@/5entities/health';
import type { BatchPredictResult, CombinedPredictionResponse } from '@/5entities/prediction';
import type { XLSXConversionResult } from '@/5entities/xlsx-file';

import type { APIResponse, MetaResponse } from './types';

// Client components hit the relative `/api` proxy (same-origin — works for
// external access; next.config rewrites it to the backend).
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
// Server Actions run on the Next server, which has no origin to resolve the
// relative `/api` proxy against (a relative URL throws "Invalid URL"). They hit
// the backend directly via BACKEND_ORIGIN (the same env next.config proxies to).
const SERVER_BACKEND_URL = process.env.BACKEND_ORIGIN || 'http://localhost:8000';

// Pick the base by execution context so callers pass a relative path only and
// no function has to opt into server-vs-client by hand (which silently breaks
// the next server-side caller). `window` is undefined on the Next server.
function backendBase(): string {
  return typeof window === 'undefined' ? SERVER_BACKEND_URL : BACKEND_URL;
}

async function apiRequest<T>(
  path: string,
  options: RequestInit,
  errorContext: string,
): Promise<APIResponse<T>> {
  try {
    const response = await fetch(`${backendBase()}${path}`, options);

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const err = await response.json();
        detail = (err as { detail?: string }).detail ?? detail;
      } catch {
        // ignore parse error on error body
      }
      return { success: false, data: null, error: `${errorContext}: ${detail}` };
    }

    const data = await response.json() as T;
    return { success: true, data, error: null };
  } catch (error) {
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : `${errorContext}: 네트워크 오류`,
    };
  }
}

export async function checkHealth(): Promise<APIResponse<HealthStatus>> {
  const base = backendBase();
  try {
    const response = await fetch(`${base}/health`, { method: 'GET' });
    const text = await response.text();
    if (!text) {
      return { success: false, data: null, error: `빈 응답 HTTP ${response.status} (${base})` };
    }
    const data = JSON.parse(text) as HealthStatus;
    return { success: true, data, error: null };
  } catch (error) {
    return {
      success: false,
      data: null,
      error: `${error instanceof Error ? error.message : '알 수 없는 오류'} (${base})`,
    };
  }
}

export async function convertXlsxToJson(
  file: File,
): Promise<APIResponse<XLSXConversionResult>> {
  const formData = new FormData();
  formData.append('file', file);
  return apiRequest(
    '/convert/xlsx-to-json',
    { method: 'POST', body: formData },
    'XLSX 변환 실패',
  );
}

export async function predictSingleCode(
  input: BIMObject,
  n = 5,
  version?: string,
): Promise<APIResponse<CombinedPredictionResponse>> {
  const { name: _name, ...attribute } = input;
  const qs = version ? `?version=${encodeURIComponent(version)}` : '';
  return apiRequest(
    `/predict${qs}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attribute, n }),
    },
    '예측 실패',
  );
}

export async function batchPredictCode(
  inputs: BIMObject[],
  n = 5,
  version?: string,
): Promise<APIResponse<BatchPredictResult>> {
  const objects = inputs.map(({ name: _name, ...attr }) => attr);
  const qs = version ? `?version=${encodeURIComponent(version)}` : '';
  return apiRequest(
    `/batch-predict${qs}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ objects, n }),
    },
    '배치 예측 실패',
  );
}

export async function fetchBimAttributes(
  page = 1,
  pageSize = 20,
  version?: string,
): Promise<APIResponse<BIMAttributeListResponse>> {
  const vq = version ? `&version=${encodeURIComponent(version)}` : '';
  return apiRequest(
    `/bim-attributes?page=${page}&page_size=${pageSize}${vq}`,
    { method: 'GET' },
    'BIM 속성 목록 조회 실패',
  );
}

export async function fetchVersions(): Promise<
  APIResponse<DbVersionListResponse>
> {
  return apiRequest(
    '/versions',
    { method: 'GET' },
    '버전 목록 조회 실패',
  );
}

export async function fetchMeta(): Promise<APIResponse<MetaResponse>> {
  return apiRequest(
    '/meta',
    { method: 'GET' },
    '메타 조회 실패',
  );
}
