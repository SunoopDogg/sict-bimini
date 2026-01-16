import type { BIMObjectInput } from '@/src/5entities/bim-object';

export interface APIResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

export interface PredictionResult {
  predicted_code: string | null;
  reasoning: string;
  confidence: number;
}

export interface BatchItemResult {
  input: BIMObjectInput;
  prediction: PredictionResult | null;
  error: string | null;
}

export interface BatchPredictResult {
  results: BatchItemResult[];
  total: number;
  successful: number;
  failed: number;
}
