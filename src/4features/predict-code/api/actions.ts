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
import { convertXlsxToJson } from '@/src/6shared/api';
import { toErrorResponse } from '@/src/6shared/lib/toErrorResponse';

import type { JsonReadResult, XlsxFileListResult, XlsxUploadActionResult } from './types';

const XLSX_DIR = join(process.cwd(), 'data', 'xlsx');
const JSON_DIR = join(process.cwd(), 'data', 'json');

function ensureDirectoryExists(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

function isValidXlsxFileName(name: string): boolean {
  return !!name && !name.includes('/') && !name.includes('\\') && name.endsWith('.xlsx');
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
    const jsonFilePath = join(JSON_DIR, basename(file.name, '.xlsx') + '.json');

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

interface RawBIMJsonProperties {
  Category?: string;
  카테고리?: string;
  'Family Name'?: string;
  '패밀리 이름'?: string;
  Family?: string;
  패밀리?: string;
  Type?: string;
  유형?: string;
  'Type Id'?: string;
  '유형 ID'?: string;
  조달청표준공사코드?: string;
  'KBIMS-부위코드'?: string;
}

interface RawBIMJsonEntry {
  IFCType?: string;
  Name?: string;
  Other?: RawBIMJsonProperties;
  기타?: RawBIMJsonProperties;
}

function mapRawToBIMObject(obj: RawBIMJsonEntry): BIMObjectInput {
  const properties = obj.Other ?? obj.기타;
  return {
    name: obj.Name ?? '',
    object_type: obj.IFCType ?? '',
    category: properties?.Category ?? properties?.카테고리 ?? '',
    family_name: properties?.['Family Name'] ?? properties?.['패밀리 이름'] ?? '',
    family: properties?.Family ?? properties?.패밀리 ?? '',
    type: properties?.Type ?? properties?.유형 ?? '',
    type_id: properties?.['Type Id'] ?? properties?.['유형 ID'] ?? '',
    pps_code: properties?.조달청표준공사코드 ?? '',
    kbims_code: properties?.['KBIMS-부위코드'] ?? '',
  };
}

export async function readJsonFileAction(
  xlsxFileName: string,
): Promise<JsonReadResult> {
  try {
    if (!isValidXlsxFileName(xlsxFileName)) {
      return { success: false, data: null, error: '유효하지 않은 파일명입니다.' };
    }

    ensureDirectoryExists(JSON_DIR);

    const jsonFileName = basename(xlsxFileName, '.xlsx') + '.json';
    const jsonFilePath = join(JSON_DIR, jsonFileName);

    if (!existsSync(jsonFilePath)) {
      return {
        success: false,
        data: null,
        error: `JSON 파일을 찾을 수 없습니다: ${jsonFileName}`,
      };
    }

    const jsonContent = readFileSync(jsonFilePath, 'utf-8');
    const rawObjects: RawBIMJsonEntry[] = JSON.parse(jsonContent);

    if (!Array.isArray(rawObjects)) {
      return {
        success: false,
        data: null,
        error: 'JSON 파일 형식이 올바르지 않습니다. 배열이 필요합니다.',
      };
    }

    const bimObjects = rawObjects.map(mapRawToBIMObject);

    return { success: true, data: bimObjects, error: null };
  } catch (error) {
    return toErrorResponse(error, 'JSON 파일 읽기에 실패했습니다.');
  }
}
