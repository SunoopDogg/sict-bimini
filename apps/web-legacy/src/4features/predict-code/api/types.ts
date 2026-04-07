import type { PredictionSession } from '@/src/5entities/prediction';
import type { APIResponse } from '@/src/6shared/api/types';

export type PredictionSaveResult = APIResponse<null>;
export type PredictionLoadResult = APIResponse<Record<string, PredictionSession[]>>;
