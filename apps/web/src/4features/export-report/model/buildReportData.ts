import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import {
  findSessionForVersion,
  getSelectedPrediction,
} from '@/5entities/prediction';

import { computeReportStats } from './computeReportStats';
import type { ReportData, ReportObjectRow } from './types';

export function buildReportData(
  objects: BIMObject[],
  predictionMap: Record<string, PredictionSession[]>,
  opts: {
    version?: string;
    fileName?: string;
    llmModel: string;
    embeddingModel: string;
    generatedAt: string;
  },
): ReportData {
  const rows: ReportObjectRow[] = objects.map((object, i) => {
    const sessions = predictionMap[i] ?? [];
    const session = findSessionForVersion(sessions, opts.version);
    if (!session) {
      return {
        object,
        session: null,
        finalKbims: null,
        finalPps: null,
        kbimsConfidence: null,
        ppsConfidence: null,
      };
    }
    const sel = getSelectedPrediction(session);
    return {
      object,
      session,
      finalKbims: sel.kbims_code,
      finalPps: sel.pps_code,
      kbimsConfidence: sel.kbims_confidence,
      ppsConfidence: sel.pps_confidence,
    };
  });

  const stats = computeReportStats(rows);

  return {
    meta: {
      version: opts.version,
      fileName: opts.fileName,
      llmModel: opts.llmModel,
      embeddingModel: opts.embeddingModel,
      generatedAt: opts.generatedAt,
    },
    stats,
    rows,
  };
}
