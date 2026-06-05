import type { PredictionSession } from '../model/types';

/**
 * Does a session belong to the given DB version?
 *
 * An `undefined` version matches any session (no filter applied). Single source
 * of truth for the version-match rule — used by `findSessionForVersion` and by
 * the prediction panel's scoped session list, so the two never drift.
 */
export function sessionMatchesVersion(
  session: PredictionSession,
  version: string | undefined,
): boolean {
  return !version || session.prediction.version === version;
}
