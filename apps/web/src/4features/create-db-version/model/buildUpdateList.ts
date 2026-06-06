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
  globalIndex: number;
  name: string;
  source: UpdateSource;
  item: VersionCreateItem;
}

function identityKey(o: BIMObject): string {
  return [o.ifc_type, o.category, o.family_name, o.family, o.type, o.type_id].join(
    '|',
  );
}

/**
 * Rows to write into the new DB version. An object is included when its
 * active-version session is a user-input card (source 'user'), OR its index is
 * in `manualAdded` (source 'confidence'). `dismissed` indices are excluded.
 * Codes come from the selected prediction (user card → userCandidate). Dedup by
 * BIM identity (user wins over confidence). Rows with no code on either side
 * are dropped (backend rejects empty records).
 */
export function buildUpdateList(
  objects: BIMObject[],
  predictionMap: Record<string, PredictionSession[]>,
  selectedVersion: string | undefined,
  opts: { manualAdded: Set<number>; dismissed: Set<number> },
): UpdateRow[] {
  const byIdentity = new Map<string, UpdateRow>();

  objects.forEach((obj, i) => {
    if (opts.dismissed.has(i)) return;
    const session = findSessionForVersion(predictionMap[i] ?? [], selectedVersion);
    if (!session) return;

    const isUserCard =
      session.selectedIndex >= getPairCount(session.prediction) &&
      session.userCandidate !== undefined;
    const source: UpdateSource | null = isUserCard
      ? 'user'
      : opts.manualAdded.has(i)
        ? 'confidence'
        : null;
    if (source === null) return;

    const sel = getSelectedPrediction(session);
    const kbims_code = sel.kbims_code ?? '';
    const pps_code = sel.pps_code ?? '';
    if (kbims_code === '' && pps_code === '') return;

    const row: UpdateRow = {
      globalIndex: i,
      name: obj.name ?? '',
      source,
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

    const key = identityKey(obj);
    const existing = byIdentity.get(key);
    // user > confidence on identity collision
    if (!existing || (existing.source === 'confidence' && source === 'user')) {
      byIdentity.set(key, row);
    }
  });

  return [...byIdentity.values()].sort((a, b) => a.globalIndex - b.globalIndex);
}
