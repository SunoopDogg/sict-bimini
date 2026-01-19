'use server';

import { predictCode, submitFeedback } from '@/src/6shared/api';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type {
  PredictionResponse,
  FeedbackResponse,
} from '@/src/5entities/prediction';

export type PredictState = {
  success: boolean;
  data?: PredictionResponse;
  error?: string;
  input?: BIMObjectInput;
};

export async function predictAction(
  _prevState: PredictState | null,
  formData: FormData
): Promise<PredictState> {
  const input: BIMObjectInput = {
    ifc_type: formData.get('ifc_type') as string,
    category: formData.get('category') as string,
    family_name: formData.get('family_name') as string,
    family: formData.get('family') as string,
    type: formData.get('type') as string,
    type_id: formData.get('type_id') as string,
  };

  try {
    const data = await predictCode(input);
    return { success: true, data, input };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : '예측에 실패했습니다.',
      input,
    };
  }
}

export type FeedbackState = {
  success: boolean;
  message?: string;
  error?: string;
};

export async function feedbackAction(
  _prevState: FeedbackState | null,
  formData: FormData
): Promise<FeedbackState> {
  const request_id = formData.get('request_id') as string;
  const correct_code = formData.get('correct_code') as string;
  const bim_data = JSON.parse(formData.get('bim_data') as string);

  try {
    const result = await submitFeedback({
      request_id,
      correct_code,
      bim_data,
    });
    return { success: true, message: result.message };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error ? error.message : '피드백 제출에 실패했습니다.',
    };
  }
}
