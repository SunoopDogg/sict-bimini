'use server';

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'fs';
import { join } from 'path';

import type { SelectionFileData, SelectionFileInfo, UserSelection } from '@/src/5entities/prediction';
import type { APIResponse } from '@/src/6shared/api/types';
import { SELECTIONS_DIR, ensureDirectoryExists } from '@/src/6shared/lib/file-utils';
import { toErrorResponse } from '@/src/6shared/lib/toErrorResponse';

const SELECTIONS_FILE = 'user-selections.json';

export async function saveUserSelectionsAction(
  selections: UserSelection[],
): Promise<APIResponse<null>> {
  try {
    ensureDirectoryExists(SELECTIONS_DIR);
    const filePath = join(SELECTIONS_DIR, SELECTIONS_FILE);

    let createdAt = new Date().toISOString();
    if (existsSync(filePath)) {
      const existing = JSON.parse(readFileSync(filePath, 'utf-8')) as SelectionFileData;
      createdAt = existing.createdAt;
    }

    const data: SelectionFileData = {
      items: selections,
      createdAt,
      modifiedAt: new Date().toISOString(),
    };

    writeFileSync(filePath, JSON.stringify(data, null, 2));
    return { success: true, data: null, error: null };
  } catch (error) {
    return toErrorResponse(error, '사용자 선택 저장에 실패했습니다.');
  }
}

export async function loadUserSelectionsAction(): Promise<APIResponse<UserSelection[]>> {
  try {
    ensureDirectoryExists(SELECTIONS_DIR);
    const filePath = join(SELECTIONS_DIR, SELECTIONS_FILE);

    if (!existsSync(filePath)) {
      return { success: true, data: [], error: null };
    }

    const raw = readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw) as SelectionFileData;
    return { success: true, data: data.items, error: null };
  } catch (error) {
    return toErrorResponse(error, '사용자 선택 불러오기에 실패했습니다.');
  }
}

export async function listSelectionFilesAction(): Promise<APIResponse<SelectionFileInfo[]>> {
  try {
    ensureDirectoryExists(SELECTIONS_DIR);
    const files = readdirSync(SELECTIONS_DIR)
      .filter((name) => name.endsWith('.json'))
      .map((name) => {
        const filePath = join(SELECTIONS_DIR, name);
        const stats = statSync(filePath);
        const raw = readFileSync(filePath, 'utf-8');
        const data = JSON.parse(raw) as SelectionFileData;
        return {
          name,
          path: filePath,
          itemCount: data.items.length,
          modifiedAt: stats.mtime.toISOString(),
        };
      })
      .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt));

    return { success: true, data: files, error: null };
  } catch (error) {
    return toErrorResponse(error, '선택 파일 목록 조회에 실패했습니다.');
  }
}
