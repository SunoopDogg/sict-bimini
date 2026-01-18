export interface Prediction {
  rank: number;
  code: string;
  description: string;
  confidence: number;
  reasoning: string;
}

export interface PredictionResponse {
  predictions: Prediction[];
  request_id: string;
}

export interface FeedbackRequest {
  request_id: string;
  correct_code: string;
  bim_data: {
    ifc_type: string;
    category: string;
    family_name: string;
    family: string;
    type: string;
    type_id: string;
  };
}

export interface FeedbackResponse {
  success: boolean;
  message: string;
}
