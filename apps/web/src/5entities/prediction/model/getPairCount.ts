import type { CombinedPredictionResponse } from './types';

/**
 * Number of aligned kbims/pps candidate pairs in a prediction.
 * Pure domain selector; the user-card index equals this value.
 */
export function getPairCount(
  prediction: CombinedPredictionResponse | null | undefined,
): number {
  if (!prediction) return 0;
  return Math.min(
    prediction.kbims.candidates.length,
    prediction.pps.candidates.length,
  );
}
