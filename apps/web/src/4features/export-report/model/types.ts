import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';

export interface ReportMeta {
  version?: string;
  fileName?: string;
  llmModel: string;
  embeddingModel: string;
  generatedAt: string; // 표시용 KST 문자열
  objectCount: number;
  predictedCount: number;
}

export interface ReportObjectRow {
  object: BIMObject;
  session: PredictionSession | null; // null = 미예측
  finalKbims: string | null;
  finalPps: string | null;
  kbimsConfidence: number | null;
  ppsConfidence: number | null;
}

export interface ReportData {
  meta: ReportMeta;
  rows: ReportObjectRow[];
}
