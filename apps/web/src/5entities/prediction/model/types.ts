import type { BIMObject } from '@/5entities/bim-object';

export type PredictionMode = 'strong' | 'weak';

export interface PredictionCandidate {
  code: string;
  llm_confidence: number;          // 0–1
  retrieval_score: number | null;
  source: 'neighbor' | 'generated';
  reasoning: string | null;
}

export interface PredictionResponse {
  target: 'kbims_code' | 'pps_code';
  mode: PredictionMode;
  candidates: PredictionCandidate[];
  low_confidence_context: boolean;
  pool_size: number;
  retrieved_k: number;
}

export interface CombinedPredictionResponse {
  kbims: PredictionResponse;
  pps: PredictionResponse;
}

export interface UserCandidate {
  kbims_code: string;
  pps_code: string;
  reasoning?: string;
}

export interface PredictionSession {
  prediction: CombinedPredictionResponse;
  selectedIndex: number;    // index into kbims/pps candidate pairs; equals pairCount when user card selected
  userCandidate?: UserCandidate;
  predicted_at: string;     // client-stamped ISO string
  dbVersion?: string;       // DB version (Qdrant collection) this prediction was run against
}

export interface UserSelection {
  objectIndex: number;
  objectName?: string;
  sessionIndex: number;
  kbims_code: string;
  pps_code: string;
  kbims_confidence: number;
  pps_confidence: number;
  object: BIMObject;
  selectedAt: string;
}

export interface SelectionFileInfo {
  name: string;
  path: string;
  itemCount: number;
  modifiedAt: string;
}

export interface SelectionFileData {
  items: UserSelection[];
  createdAt: string;
  modifiedAt: string;
}

export interface BatchItemResult {
  input: BIMObject;
  prediction: CombinedPredictionResponse | null;
  error: string | null;
}

export interface BatchPredictResult {
  results: BatchItemResult[];
  total: number;
  successful: number;
  failed: number;
}
