import { existsSync, mkdirSync } from 'fs';
import { basename, join } from 'path';

export const XLSX_DIR = join(process.cwd(), 'data', 'xlsx');
export const JSON_DIR = join(process.cwd(), 'data', 'json');
export const PREDICTIONS_DIR = join(process.cwd(), 'data', 'predictions');
export const SELECTIONS_DIR = join(process.cwd(), 'data', 'selections');

export function ensureDirectoryExists(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

export function isValidXlsxFileName(name: string): boolean {
  return (
    !!name &&
    !name.includes('/') &&
    !name.includes('\\') &&
    name.endsWith('.xlsx')
  );
}

export function deriveJsonPath(dir: string, xlsxFileName: string): string {
  return join(dir, basename(xlsxFileName, '.xlsx') + '.json');
}
