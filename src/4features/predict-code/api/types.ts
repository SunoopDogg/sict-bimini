import type { BatchPredictResult } from '@/src/5entities/prediction';
import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';

export type UploadStatus = 'idle' | 'uploading' | 'done';

export type XlsxFileListResult = {
  success: boolean;
  files?: XlsxFileInfo[];
  error?: string;
};

export type XlsxUploadResult = {
  success: boolean;
  file?: XlsxFileInfo;
  jsonPath?: string;
  totalObjects?: number;
  error?: string;
};

export type BatchPredictionResult = {
  success: boolean;
  data?: BatchPredictResult;
  error?: string;
};
