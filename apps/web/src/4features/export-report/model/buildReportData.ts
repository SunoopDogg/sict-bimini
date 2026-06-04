import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import { getPairCount } from '@/5entities/prediction';

import type { ReportData, ReportObjectRow } from './types';

// Pick the session to report for an object: the latest one predicted against
// the report's DB version, so an A-DB report never shows B-DB results (and vice
// versa). Falls back to the latest session when no version is given.
function pickSession(
  sessions: PredictionSession[],
  version?: string,
): PredictionSession | null {
  if (sessions.length === 0) return null;
  if (!version) return sessions[sessions.length - 1];
  for (let i = sessions.length - 1; i >= 0; i--) {
    if (sessions[i].prediction.version === version) return sessions[i];
  }
  return null;
}

function resolveFinal(session: PredictionSession): {
  finalKbims: string | null;
  finalPps: string | null;
  kbimsConfidence: number | null;
  ppsConfidence: number | null;
} {
  const pred = session.prediction;
  const pairCount = getPairCount(pred);
  if (session.selectedIndex < pairCount) {
    const k = pred.kbims.candidates[session.selectedIndex];
    const p = pred.pps.candidates[session.selectedIndex];
    return {
      finalKbims: k?.code ?? null,
      finalPps: p?.code ?? null,
      kbimsConfidence: k?.llm_confidence ?? null,
      ppsConfidence: p?.llm_confidence ?? null,
    };
  }
  // 사용자 수동 입력 카드 (예측 신뢰도 없음)
  return {
    finalKbims: session.userCandidate?.kbims_code ?? null,
    finalPps: session.userCandidate?.pps_code ?? null,
    kbimsConfidence: null,
    ppsConfidence: null,
  };
}

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
    const session = pickSession(sessions, opts.version);
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
    return { object, session, ...resolveFinal(session) };
  });

  return {
    meta: {
      version: opts.version,
      fileName: opts.fileName,
      llmModel: opts.llmModel,
      embeddingModel: opts.embeddingModel,
      generatedAt: opts.generatedAt,
      objectCount: rows.length,
      predictedCount: rows.filter((r) => r.session !== null).length,
    },
    rows,
  };
}
