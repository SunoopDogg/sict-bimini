import { classifyPredictionMatch } from '@/5entities/prediction';

import type { ReportObjectRow, ReportStats } from './types';

/**
 * Aggregate prediction-quality stats over the report rows. Only predicted rows
 * (session !== null) contribute. Accuracy denominators count only rows that
 * have ground truth (object.kbims_code / pps_code); average confidence skips
 * user-input cards (no model score). Pure — shown between the report meta
 * header and the per-object summary table.
 */
export function computeReportStats(rows: ReportObjectRow[]): ReportStats {
  let predictedCount = 0;
  let kbimsJudgeable = 0;
  let kbimsCorrect = 0;
  let ppsJudgeable = 0;
  let ppsCorrect = 0;
  let kbimsConfSum = 0;
  let kbimsConfN = 0;
  let ppsConfSum = 0;
  let ppsConfN = 0;

  for (const r of rows) {
    if (!r.session) continue;
    predictedCount++;

    if (r.object.kbims_code) {
      kbimsJudgeable++;
      if (classifyPredictionMatch(r.finalKbims, r.object.kbims_code) === 'match')
        kbimsCorrect++;
    }
    if (r.object.pps_code) {
      ppsJudgeable++;
      if (classifyPredictionMatch(r.finalPps, r.object.pps_code) === 'match')
        ppsCorrect++;
    }

    if (r.kbimsConfidence !== null) {
      kbimsConfSum += r.kbimsConfidence;
      kbimsConfN++;
    }
    if (r.ppsConfidence !== null) {
      ppsConfSum += r.ppsConfidence;
      ppsConfN++;
    }
  }

  return {
    objectCount: rows.length,
    predictedCount,
    kbimsJudgeable,
    kbimsCorrect,
    ppsJudgeable,
    ppsCorrect,
    avgKbimsConfidence: kbimsConfN ? kbimsConfSum / kbimsConfN : null,
    avgPpsConfidence: ppsConfN ? ppsConfSum / ppsConfN : null,
  };
}
