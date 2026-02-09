import type { BIMObjectInput } from '@/src/5entities/bim-object';

export interface PredictionResult {
  predicted_code: string | null;
  predicted_pps_code: string | null;
  reasoning: string;
  confidence: number;
  predicted_at: string;
}

export interface PredictionCandidates {
  predictions: PredictionResult[];
}

export interface PredictionSession {
  candidates: PredictionResult[];
  userCandidate?: PredictionResult;
  selectedIndex: number;
  predicted_at: string;
}

export interface UserSelection {
  objectIndex: number;
  objectName: string;
  sessionIndex: number;
  candidate: PredictionResult;
  object: BIMObjectInput;
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

interface BatchItemResult {
  input: BIMObjectInput;
  prediction: PredictionCandidates | null;
  error: string | null;
}

export interface BatchPredictResult {
  results: BatchItemResult[];
  total: number;
  successful: number;
  failed: number;
}
