import type {
  CombinedPredictionResponse,
  PredictionSession,
  UserSelection,
} from '../model/types';

/**
 * A prediction response with no candidates — used for sessions reconstructed
 * from saved user selections, where only the user's chosen codes are known.
 */
export const EMPTY_COMBINED_PREDICTION_RESPONSE: CombinedPredictionResponse = {
  kbims: {
    target: 'kbims_code',
    mode: 'strong',
    candidates: [],
    low_confidence_context: false,
    pool_size: 0,
    retrieved_k: 0,
  },
  pps: {
    target: 'pps_code',
    mode: 'strong',
    candidates: [],
    low_confidence_context: false,
    pool_size: 0,
    retrieved_k: 0,
  },
};

/**
 * Rebuild the per-object prediction map from saved user selections.
 * Each selection becomes a single "user card" session (pairCount=0, so
 * selectedIndex=0 marks the user-entered codes as selected).
 */
export function buildSelectionSessionMap(
  selections: UserSelection[],
): Record<string, PredictionSession[]> {
  const map: Record<string, PredictionSession[]> = {};
  selections.forEach((sel, i) => {
    map[i] = [
      {
        prediction: EMPTY_COMBINED_PREDICTION_RESPONSE,
        userCandidate: { kbims_code: sel.kbims_code, pps_code: sel.pps_code },
        selectedIndex: 0,
        predicted_at: sel.selectedAt,
      },
    ];
  });
  return map;
}
