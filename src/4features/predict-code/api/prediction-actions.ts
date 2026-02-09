'use server';

import { existsSync, readFileSync, writeFileSync } from 'fs';

import type { PredictionResult } from '@/src/5entities/prediction';
import { toErrorResponse } from '@/src/6shared/lib/toErrorResponse';

import type { PredictionLoadResult, PredictionSaveResult } from './types';
import {
  PREDICTIONS_DIR,
  deriveJsonPath,
  ensureDirectoryExists,
  isValidXlsxFileName,
} from '@/src/6shared/lib/file-utils';

export async function savePredictionsAction(
  xlsxFileName: string,
  predictions: Record<string, PredictionResult[]>,
): Promise<PredictionSaveResult> {
  try {
    if (!isValidXlsxFileName(xlsxFileName)) {
      return { success: false, data: null, error: '유효하지 않은 파일명입니다.' };
    }

    ensureDirectoryExists(PREDICTIONS_DIR);

    const predFilePath = deriveJsonPath(PREDICTIONS_DIR, xlsxFileName);

    writeFileSync(predFilePath, JSON.stringify(predictions, null, 2));

    return { success: true, data: null, error: null };
  } catch (error) {
    return toErrorResponse(error, '예측 결과 저장에 실패했습니다.');
  }
}

export async function loadPredictionsAction(
  xlsxFileName: string,
): Promise<PredictionLoadResult> {
  try {
    if (!isValidXlsxFileName(xlsxFileName)) {
      return { success: false, data: null, error: '유효하지 않은 파일명입니다.' };
    }

    ensureDirectoryExists(PREDICTIONS_DIR);

    const predFilePath = deriveJsonPath(PREDICTIONS_DIR, xlsxFileName);

    if (!existsSync(predFilePath)) {
      return { success: true, data: {}, error: null };
    }

    const content = readFileSync(predFilePath, 'utf-8');
    const predictions: Record<string, PredictionResult[]> = JSON.parse(content);

    return { success: true, data: predictions, error: null };
  } catch (error) {
    return toErrorResponse(error, '예측 결과 불러오기에 실패했습니다.');
  }
}
