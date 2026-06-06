import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import {
  findSessionForVersion,
  getSelectedPrediction,
} from '@/5entities/prediction';

/**
 * Object indices to bulk-add: GT-less objects whose selected prediction has
 * BOTH kbims and pps confidence >= threshold. Pure — drives the
 * "신뢰도 ≥N% 추가" button. thresholdPercent is 0–100 (model conf is 0–1).
 */
export function selectByConfidence(
  objects: BIMObject[],
  predictionMap: Record<string, PredictionSession[]>,
  selectedVersion: string | undefined,
  thresholdPercent: number,
): Set<number> {
  const out = new Set<number>();
  const th = thresholdPercent / 100;
  objects.forEach((obj, i) => {
    // GT 있는 객체 제외 (이미 DB에 정답 존재)
    if (obj.kbims_code !== '' || obj.pps_code !== '') return;
    const session = findSessionForVersion(predictionMap[i] ?? [], selectedVersion);
    if (!session) return;
    const sel = getSelectedPrediction(session);
    if (sel.kbims_confidence === null || sel.pps_confidence === null) return;
    if (sel.kbims_confidence >= th && sel.pps_confidence >= th) out.add(i);
  });
  return out;
}
