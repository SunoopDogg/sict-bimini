import type { PredictionSession } from '../model/types';
import { getPairCount } from './getPairCount';

export interface SelectedPrediction {
  kbims_code: string | null;
  pps_code: string | null;
  kbims_confidence: number | null;
  pps_confidence: number | null;
}

/**
 * Resolve a session's selected KBIMS/PPS codes and confidences.
 *
 * `selectedIndex < pairCount` means a predicted candidate pair is selected, so
 * codes/confidences come from that pair. Otherwise the user-entered card is
 * selected: codes come from `userCandidate` and confidence is null (no model
 * score). Single source of truth for "what did the user end up picking" —
 * shared by the report, the object-list status icons, and selection saving.
 */
export function getSelectedPrediction(
  session: PredictionSession,
): SelectedPrediction {
  const pred = session.prediction;
  const pairCount = getPairCount(pred);
  if (session.selectedIndex < pairCount) {
    const k = pred.kbims.candidates[session.selectedIndex];
    const p = pred.pps.candidates[session.selectedIndex];
    return {
      kbims_code: k?.code ?? null,
      pps_code: p?.code ?? null,
      kbims_confidence: k?.llm_confidence ?? null,
      pps_confidence: p?.llm_confidence ?? null,
    };
  }
  return {
    kbims_code: session.userCandidate?.kbims_code ?? null,
    pps_code: session.userCandidate?.pps_code ?? null,
    kbims_confidence: null,
    pps_confidence: null,
  };
}
