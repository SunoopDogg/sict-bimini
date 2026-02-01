export interface PredictionResult {
  predicted_code: string | null;
  reasoning: string;
  confidence: number;
  predicted_at?: string;
}

interface BatchItemResult {
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
