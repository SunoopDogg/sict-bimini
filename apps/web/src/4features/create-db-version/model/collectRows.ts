import type { BIMObject } from '@/5entities/bim-object';
import type { VersionCreateItem } from '@/5entities/db-version';
import type { PredictionSession } from '@/5entities/prediction';
import {
  findSessionForVersion,
  getPairCount,
  getSelectedPrediction,
} from '@/5entities/prediction';

export type UpdateSource = 'user' | 'confidence';

export interface UpdateRow {
  /** Stable key = the 6 BIM identity fields joined. Dedup across files/DBs. */
  identityKey: string;
  name: string;
  source: UpdateSource;
  /** DB version (Qdrant collection) the prediction was run against. */
  version: string;
  /** Confidence threshold (%) at add-time for 'confidence' rows; null for 'user'. */
  threshold: number | null;
  item: VersionCreateItem;
}

export function identityKey(o: BIMObject): string {
  return [o.ifc_type, o.category, o.family_name, o.family, o.type, o.type_id].join(
    '|',
  );
}

function snapshot(
  obj: BIMObject,
  session: PredictionSession,
  source: UpdateSource,
  threshold: number | null,
): UpdateRow | null {
  const sel = getSelectedPrediction(session);
  const kbims_code = sel.kbims_code ?? '';
  const pps_code = sel.pps_code ?? '';
  // Backend rejects records with no code on either side.
  if (kbims_code === '' && pps_code === '') return null;
  return {
    identityKey: identityKey(obj),
    name: obj.name ?? '',
    source,
    version: session.prediction.version,
    threshold,
    item: {
      ifc_type: obj.ifc_type,
      category: obj.category,
      family_name: obj.family_name,
      family: obj.family,
      type: obj.type,
      type_id: obj.type_id,
      kbims_code,
      pps_code,
    },
  };
}

/**
 * Snapshot rows for GT-less objects (both codes empty) whose active-source
 * session has BOTH kbims and pps confidence >= thresholdPercent. The caller
 * merges these into a persistent accumulator, so they survive source changes.
 */
export function collectConfidenceRows(
  objects: BIMObject[],
  predictionMap: Record<string, PredictionSession[]>,
  sourceVersion: string | undefined,
  thresholdPercent: number,
): UpdateRow[] {
  const out: UpdateRow[] = [];
  const th = thresholdPercent / 100;
  objects.forEach((obj, i) => {
    if (obj.kbims_code !== '' || obj.pps_code !== '') return; // GT exists → skip
    const session = findSessionForVersion(predictionMap[i] ?? [], sourceVersion);
    if (!session) return;
    const sel = getSelectedPrediction(session);
    if (sel.kbims_confidence === null || sel.pps_confidence === null) return;
    if (sel.kbims_confidence < th || sel.pps_confidence < th) return;
    const row = snapshot(obj, session, 'confidence', thresholdPercent);
    if (row) out.push(row);
  });
  return out;
}

/**
 * Snapshot rows for objects whose active-source session is a user-input card
 * (selectedIndex past the predicted pairs, with a userCandidate). Loaded on
 * demand so they too survive source changes.
 */
export function collectUserRows(
  objects: BIMObject[],
  predictionMap: Record<string, PredictionSession[]>,
  sourceVersion: string | undefined,
): UpdateRow[] {
  const out: UpdateRow[] = [];
  objects.forEach((obj, i) => {
    const session = findSessionForVersion(predictionMap[i] ?? [], sourceVersion);
    if (!session) return;
    const isUserCard =
      session.selectedIndex >= getPairCount(session.prediction) &&
      session.userCandidate !== undefined;
    if (!isUserCard) return;
    const row = snapshot(obj, session, 'user', null);
    if (row) out.push(row);
  });
  return out;
}
