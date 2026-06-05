import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';

export interface ReportMeta {
  version?: string;
  fileName?: string;
  llmModel: string;
  embeddingModel: string;
  generatedAt: string; // 표시용 KST 문자열
}

export interface ReportObjectRow {
  object: BIMObject;
  session: PredictionSession | null; // null = 미예측
  finalKbims: string | null;
  finalPps: string | null;
  kbimsConfidence: number | null;
  ppsConfidence: number | null;
}

export interface ReportStats {
  objectCount: number;
  predictedCount: number;
  // 정답(object.kbims_code/pps_code) 보유 + 예측된 객체만 분모로 잡은 정확도
  kbimsJudgeable: number;
  kbimsCorrect: number;
  ppsJudgeable: number;
  ppsCorrect: number;
  // 모델 점수 있는 선택(예측 후보)만 평균; 사용자 입력 카드는 제외 → null 가능
  avgKbimsConfidence: number | null;
  avgPpsConfidence: number | null;
}

export interface ReportData {
  meta: ReportMeta;
  stats: ReportStats;
  rows: ReportObjectRow[];
}
