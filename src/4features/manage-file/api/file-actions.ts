'use server';

import {
  existsSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'fs';
import { join } from 'path';

import { convertXlsxToJson } from '@/src/6shared/api';
import {
  JSON_DIR,
  XLSX_DIR,
  deriveJsonPath,
  ensureDirectoryExists,
  isValidXlsxFileName,
} from '@/src/6shared/lib/file-utils';
import { toErrorResponse } from '@/src/6shared/lib/toErrorResponse';

import type { XlsxFileListResult, XlsxUploadActionResult } from './types';

export async function listXlsxFilesAction(): Promise<XlsxFileListResult> {
  try {
    ensureDirectoryExists(XLSX_DIR);
    const files = readdirSync(XLSX_DIR)
      .filter((name) => name.endsWith('.xlsx'))
      .map((name) => {
        const filePath = join(XLSX_DIR, name);
        const stats = statSync(filePath);
        return {
          name,
          path: filePath,
          size: stats.size,
          createdAt: stats.birthtime.toISOString(),
          modifiedAt: stats.mtime.toISOString(),
        };
      })
      .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));

    return { success: true, data: files, error: null };
  } catch (error) {
    return toErrorResponse(error, '파일 목록 조회에 실패했습니다.');
  }
}

export async function uploadAndConvertXlsxAction(
  formData: FormData,
  overwrite: boolean = true,
): Promise<XlsxUploadActionResult> {
  let xlsxFilePath: string | null = null;

  try {
    ensureDirectoryExists(XLSX_DIR);
    ensureDirectoryExists(JSON_DIR);

    const file = formData.get('file') as File;

    if (!file) {
      return { success: false, data: null, error: '파일이 선택되지 않았습니다.' };
    }

    if (!isValidXlsxFileName(file.name)) {
      return { success: false, data: null, error: 'xlsx 파일만 업로드 가능합니다.' };
    }

    xlsxFilePath = join(XLSX_DIR, file.name);
    const jsonFilePath = deriveJsonPath(JSON_DIR, file.name);

    if (existsSync(xlsxFilePath)) {
      if (!overwrite) {
        return {
          success: false,
          data: null,
          error: `'${file.name}' 파일이 이미 존재합니다.`,
        };
      }
      unlinkSync(xlsxFilePath);
      if (existsSync(jsonFilePath)) {
        unlinkSync(jsonFilePath);
      }
    }

    // 1. XLSX 파일 저장
    const buffer = await file.arrayBuffer();
    writeFileSync(xlsxFilePath, Buffer.from(buffer));

    // 2. 백엔드 API를 통해 XLSX → JSON 변환
    const conversionResult = await convertXlsxToJson(file);

    if (!conversionResult.success || !conversionResult.data) {
      if (xlsxFilePath && existsSync(xlsxFilePath)) {
        unlinkSync(xlsxFilePath);
      }
      return {
        success: false,
        data: null,
        error: conversionResult.error || 'XLSX 변환에 실패했습니다.',
      };
    }

    // 3. 변환된 JSON을 파일로 저장
    writeFileSync(
      jsonFilePath,
      JSON.stringify(conversionResult.data.objects, null, 2),
    );

    const stats = statSync(xlsxFilePath);
    return {
      success: true,
      data: {
        file: {
          name: file.name,
          path: xlsxFilePath,
          size: stats.size,
          createdAt: stats.birthtime.toISOString(),
          modifiedAt: stats.mtime.toISOString(),
        },
      },
      error: null,
    };
  } catch (error) {
    if (xlsxFilePath && existsSync(xlsxFilePath)) {
      unlinkSync(xlsxFilePath);
    }

    return toErrorResponse(error, '파일 업로드에 실패했습니다.');
  }
}
