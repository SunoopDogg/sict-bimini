'use server';

import { existsSync, readFileSync } from 'fs';
import { basename } from 'path';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import {
  JSON_DIR,
  deriveJsonPath,
  ensureDirectoryExists,
  isValidXlsxFileName,
} from '@/src/6shared/lib/file-utils';
import { toErrorResponse } from '@/src/6shared/lib/toErrorResponse';

import type { JsonReadResult } from './types';

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

    const jsonFilePath = deriveJsonPath(JSON_DIR, xlsxFileName);

    if (!existsSync(jsonFilePath)) {
      return {
        success: false,
        data: null,
        error: `JSON 파일을 찾을 수 없습니다: ${basename(xlsxFileName, '.xlsx') + '.json'}`,
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
