import type { BIMAttributeListResponse } from '@/5entities/bim-attribute';
import type { BIMObject } from '@/5entities/bim-object';
import type { DbVersionListResponse } from '@/5entities/db-version';
import type { HealthStatus } from '@/5entities/health';
import type { BatchPredictResult, CombinedPredictionResponse } from '@/5entities/prediction';
import type { XLSXConversionResult } from '@/5entities/xlsx-file';

import type { APIResponse } from './types';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

async function apiRequest<T>(
  url: string,
  options: RequestInit,
  errorContext: string,
): Promise<APIResponse<T>> {
  try {
    const response = await fetch(url, options);

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
  try {
    const response = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
    const text = await response.text();
    if (!text) {
      return { success: false, data: null, error: `빈 응답 HTTP ${response.status} (${BACKEND_URL})` };
    }
    const data = JSON.parse(text) as HealthStatus;
    return { success: true, data, error: null };
  } catch (error) {
    return {
      success: false,
      data: null,
      error: `${error instanceof Error ? error.message : '알 수 없는 오류'} (${BACKEND_URL})`,
    };
  }
}

export async function convertXlsxToJson(
  file: File,
): Promise<APIResponse<XLSXConversionResult>> {
  const formData = new FormData();
  formData.append('file', file);
  return apiRequest(
    `${BACKEND_URL}/convert/xlsx-to-json`,
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
    `${BACKEND_URL}/predict${qs}`,
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
    `${BACKEND_URL}/batch-predict${qs}`,
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
    `${BACKEND_URL}/bim-attributes?page=${page}&page_size=${pageSize}${vq}`,
    { method: 'GET' },
    'BIM 속성 목록 조회 실패',
  );
}

export async function fetchVersions(): Promise<
  APIResponse<DbVersionListResponse>
> {
  return apiRequest(
    `${BACKEND_URL}/versions`,
    { method: 'GET' },
    '버전 목록 조회 실패',
  );
}
