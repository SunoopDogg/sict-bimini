import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type {
  APIResponse,
  BatchPredictResult,
} from '@/src/5entities/prediction';
import type { XLSXConversionResult } from '@/src/5entities/xlsx-file';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

export async function convertXlsxToJson(
  file: File,
): Promise<APIResponse<XLSXConversionResult>> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${BACKEND_URL}${API_VERSION}/convert/xlsx-to-json`,
    {
      method: 'POST',
      body: formData,
    },
  );

  if (!response.ok) {
    return {
      success: false,
      data: null,
      error: `XLSX conversion failed: ${response.statusText}`,
    };
  }

  return response.json();
}

/**
 * 배치 BIM 객체에 대한 코드 예측
 */
export async function batchPredictCode(
  inputs: BIMObjectInput[],
  topK: number = 3,
): Promise<APIResponse<BatchPredictResult>> {
  const response = await fetch(`${BACKEND_URL}${API_VERSION}/batch-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      objects: inputs,
      top_k: topK,
    }),
  });

  if (!response.ok) {
    return {
      success: false,
      data: null,
      error: `Batch prediction failed: ${response.statusText}`,
    };
  }

  return response.json();
}
