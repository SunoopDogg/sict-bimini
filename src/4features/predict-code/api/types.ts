import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { APIResponse } from '@/src/6shared/api/types';
import type { XlsxFileInfo, XlsxUploadResult } from '@/src/5entities/xlsx-file';

// UI state type (kept in features layer)
export type UploadStatus = 'idle' | 'uploading' | 'done';

// APIResponse<T> pattern for consistent API responses
export type XlsxFileListResult = APIResponse<XlsxFileInfo[]>;
export type XlsxUploadActionResult = APIResponse<XlsxUploadResult>;
export type JsonReadResult = APIResponse<BIMObjectInput[]>;
