import type { BIMObject } from '@/5entities/bim-object';
import type { APIResponse } from '@/6shared/api/types';
import type { XlsxFileInfo, XlsxUploadResult } from '@/5entities/xlsx-file';

export type UploadStatus = 'idle' | 'uploading' | 'done';

export type XlsxFileListResult = APIResponse<XlsxFileInfo[]>;
export type XlsxUploadActionResult = APIResponse<XlsxUploadResult>;
export type JsonReadResult = APIResponse<BIMObject[]>;
