export interface XlsxFileInfo {
  name: string;
  path: string;
  size: number;
  createdAt: string; // ISO string format
  modifiedAt: string; // ISO string format
}

export interface XLSXConversionResult {
  objects: Record<string, unknown>[];
  total_objects: number;
  processing_time_seconds: number;
  source_filename: string;
}
