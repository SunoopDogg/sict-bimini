import type { BIMAttributeListResponse } from '@/src/5entities/bim-attribute';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { HealthStatus } from '@/src/5entities/health';
import type { BatchPredictResult, PredictionCandidates } from '@/src/5entities/prediction';
import type { XLSXConversionResult } from '@/src/5entities/xlsx-file';

import type { APIResponse } from './types';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

async function apiRequest<T>(
  url: string,
  options: RequestInit,
  errorContext: string,
): Promise<APIResponse<T>> {
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      return {
        success: false,
        data: null,
        error: `${errorContext}: ${response.statusText}`,
      };
    }

    return response.json();
  } catch (error) {
    return {
      success: false,
      data: null,
      error: error instanceof Error ? error.message : `${errorContext}: 네트워크 오류`,
    };
  }
}

export async function checkHealth(): Promise<APIResponse<HealthStatus>> {
  return apiRequest(
    `${BACKEND_URL}${API_VERSION}/health`,
    { method: 'GET' },
    '서버 상태 확인 실패',
  );
}

export async function convertXlsxToJson(
  file: File,
): Promise<APIResponse<XLSXConversionResult>> {
  const formData = new FormData();
  formData.append('file', file);

  return apiRequest(
    `${BACKEND_URL}${API_VERSION}/convert/xlsx-to-json`,
    { method: 'POST', body: formData },
    'XLSX 변환 실패',
  );
}

export async function batchPredictCode(
  inputs: BIMObjectInput[],
  topK: number = 3,
): Promise<APIResponse<BatchPredictResult>> {
  return apiRequest(
    `${BACKEND_URL}${API_VERSION}/batch-predict`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ objects: inputs, top_k: topK }),
    },
    '배치 예측 실패',
  );
}

export async function predictSingleCode(
  input: BIMObjectInput,
  topK: number = 3,
): Promise<APIResponse<PredictionCandidates>> {
  return apiRequest(
    `${BACKEND_URL}${API_VERSION}/predict?top_k=${topK}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    },
    '예측 실패',
  );
}

export async function fetchBimAttributes(
  page: number = 1,
  pageSize: number = 20,
): Promise<APIResponse<BIMAttributeListResponse>> {
  return apiRequest(
    `${BACKEND_URL}${API_VERSION}/bim-attributes?page=${page}&page_size=${pageSize}`,
    { method: 'GET' },
    'BIM 속성 목록 조회 실패',
  );
}
