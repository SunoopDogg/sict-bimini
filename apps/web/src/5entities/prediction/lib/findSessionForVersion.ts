import type { PredictionSession } from '../model/types';

/**
 * The latest session predicted against a given DB version, or null if none.
 * When `version` is undefined, falls back to the latest session of any version.
 *
 * Single source of truth for "which session belongs to this DB" — used both to
 * decide whether an object still needs predicting and to pick the session a
 * report renders, so the two never drift.
 */
export function findSessionForVersion(
  sessions: PredictionSession[],
  version: string | undefined,
): PredictionSession | null {
  if (sessions.length === 0) return null;
  if (!version) return sessions[sessions.length - 1];
  for (let i = sessions.length - 1; i >= 0; i--) {
    if (sessions[i].prediction.version === version) return sessions[i];
  }
  return null;
}
