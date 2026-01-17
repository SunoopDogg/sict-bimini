import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type {
  FeedbackRequest,
  FeedbackResponse,
  PredictionResponse,
} from '@/src/5entities/prediction';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function predictCode(
  input: BIMObjectInput,
): Promise<PredictionResponse> {
  const response = await fetch(`${BACKEND_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    throw new Error(`Prediction failed: ${response.statusText}`);
  }

  return response.json();
}

export async function submitFeedback(
  feedback: FeedbackRequest,
): Promise<FeedbackResponse> {
  const response = await fetch(`${BACKEND_URL}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback),
  });

  if (!response.ok) {
    throw new Error(`Feedback submission failed: ${response.statusText}`);
  }

  return response.json();
}
