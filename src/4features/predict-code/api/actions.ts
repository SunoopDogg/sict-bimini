'use server';

import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'fs';
import { basename, join } from 'path';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import { batchPredictCode, convertXlsxToJson } from '@/src/6shared/api';

import type {
  BatchPredictionResult,
  XlsxFileListResult,
  XlsxUploadResult,
} from './types';

const XLSX_DIR = join(process.cwd(), 'data', 'xlsx');
const JSON_DIR = join(process.cwd(), 'data', 'json');

function ensureDirectoryExists(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

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

    return { success: true, files };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : '파일 목록 조회에 실패했습니다.',
    };
  }
}

export async function uploadAndConvertXlsxAction(
  formData: FormData,
  overwrite: boolean = true,
): Promise<XlsxUploadResult> {
  let xlsxFilePath: string | null = null;

  try {
    ensureDirectoryExists(XLSX_DIR);
    ensureDirectoryExists(JSON_DIR);

    const file = formData.get('file') as File;

    if (!file) {
      return { success: false, error: '파일이 선택되지 않았습니다.' };
    }

    if (!file.name.endsWith('.xlsx')) {
      return { success: false, error: 'xlsx 파일만 업로드 가능합니다.' };
    }

    xlsxFilePath = join(XLSX_DIR, file.name);
    const jsonFilePath = join(JSON_DIR, basename(file.name, '.xlsx') + '.json');

    if (existsSync(xlsxFilePath)) {
      if (!overwrite) {
        return {
          success: false,
          error: `'${file.name}' 파일이 이미 존재합니다.`,
        };
      }
      // 덮어쓰기: 기존 파일 삭제
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
      // 롤백: XLSX 삭제
      if (xlsxFilePath && existsSync(xlsxFilePath)) {
        unlinkSync(xlsxFilePath);
      }
      return {
        success: false,
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
      file: {
        name: file.name,
        path: xlsxFilePath,
        size: stats.size,
        createdAt: stats.birthtime.toISOString(),
        modifiedAt: stats.mtime.toISOString(),
      },
      jsonPath: jsonFilePath,
      totalObjects: conversionResult.data.total_objects,
    };
  } catch (error) {
    const file = formData.get('file') as File;
    if (file) {
      const xlsxFilePath = join(XLSX_DIR, file.name);
      if (existsSync(xlsxFilePath)) {
        unlinkSync(xlsxFilePath);
      }
    }

    return {
      success: false,
      error:
        error instanceof Error ? error.message : '파일 업로드에 실패했습니다.',
    };
  }
}
