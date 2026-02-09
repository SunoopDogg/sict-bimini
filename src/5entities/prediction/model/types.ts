export type { APIResponse } from '@/src/6shared/api/types';

export interface PredictionResult {
  predicted_code: string | null;
  reasoning: string;
  confidence: number;
}

export interface BatchItemResult {
  input: {
    name: string;
    object_type: string;
    category: string;
    family_name: string;
    family: string;
    type: string;
    type_id: string;
    pps_code: string;
    kbims_code: string;
  };
  prediction: PredictionResult | null;
  error: string | null;
}

export interface BatchPredictResult {
  results: BatchItemResult[];
  total: number;
  successful: number;
  failed: number;
}
