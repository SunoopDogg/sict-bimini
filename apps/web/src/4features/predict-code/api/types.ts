import type { PredictionSession } from '@/5entities/prediction';
import type { APIResponse } from '@/6shared/api/types';

export type PredictionSaveResult = APIResponse<null>;
export type PredictionLoadResult = APIResponse<Record<string, PredictionSession[]>>;
