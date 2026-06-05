import { useState } from 'react';

import type { BIMObject } from '@/5entities/bim-object';
import type {
  BatchItemResult,
  CombinedPredictionResponse,
  PredictionSession,
  UserCandidate,
} from '@/5entities/prediction';
import { findSessionForVersion, getPairCount } from '@/5entities/prediction';
import { batchPredictCode } from '@/6shared/api';
import { useLocale } from '@/6shared/i18n';
import { savePredictionsAction } from '@/4features/predict-code';

// Objects predicted per /batch-predict request during a full-file "export".
// Kept small so progress updates often; the server processes a batch
// sequentially anyway, so smaller chunks barely change total time. Stays well
// under the API's 100-object cap (Field max_length=100).
const PREDICT_CHUNK = 5;

function toSession(prediction: CombinedPredictionResponse): PredictionSession {
  return {
    prediction,
    selectedIndex: 0,
    predicted_at: new Date().toISOString(),
  };
}

// Zip /batch-predict results back to their object indices. The endpoint
// preserves request order, so `indices[i]` is the source index of `results[i]`.
function toEntries(
  results: BatchItemResult[],
  indices: number[],
): { index: number; session: PredictionSession }[] {
  return results
    .map((item, i) =>
      item.prediction
        ? { index: indices[i], session: toSession(item.prediction) }
        : null,
    )
    .filter(
      (e): e is { index: number; session: PredictionSession } => e !== null,
    );
}

interface UsePredictionSessionsOptions {
  selectedFile: string | undefined;
  onSelectionSync: (
    objectIndex: number,
    sessionIndex: number,
    session: PredictionSession,
    action: 'add' | 'remove',
  ) => void;
}

export function usePredictionSessions({
  selectedFile,
  onSelectionSync,
}: UsePredictionSessionsOptions) {
  const { t } = useLocale();
  const [predictionMap, setPredictionMap] = useState<Record<string, PredictionSession[]>>({});

  const saveToDisk = (nextMap: Record<string, PredictionSession[]>) => {
    if (selectedFile) {
      savePredictionsAction(selectedFile, nextMap);
    }
  };

  // Returns the merged map so callers can chain appends across awaits without
  // waiting for a re-render (state closure would be stale). Pass `base` to
  // thread a running map through a loop; defaults to current state.
  const appendSessions = (
    entries: { index: number; session: PredictionSession }[],
    base: Record<string, PredictionSession[]> = predictionMap,
  ): Record<string, PredictionSession[]> => {
    const nextMap = { ...base };
    for (const { index, session } of entries) {
      const existing = nextMap[index] ?? [];
      nextMap[index] = [...existing, session];
    }
    setPredictionMap(nextMap);
    saveToDisk(nextMap);
    return nextMap;
  };

  // Predict every object that has no session for `version` yet, then return the
  // fresh map so a report can be built from it immediately (state would be
  // stale). Chunks by PREDICT_CHUNK and persists each chunk, so a later failure
  // (or cancel) keeps earlier work; `onProgress` fires after each chunk for a
  // live "done / total" count; `shouldCancel` is polled before each chunk so the
  // caller can stop at a chunk boundary (the returned map is then partial).
  const ensureAllPredicted = async (
    objects: BIMObject[],
    version: string | undefined,
    onProgress?: (done: number, total: number) => void,
    shouldCancel?: () => boolean,
  ): Promise<Record<string, PredictionSession[]>> => {
    const missing = objects
      .map((object, index) => ({ object, index }))
      .filter(
        ({ index }) =>
          findSessionForVersion(predictionMap[index] ?? [], version) === null,
      );
    if (missing.length === 0) return predictionMap;

    const total = missing.length;
    onProgress?.(0, total);

    let map = predictionMap;
    let done = 0;
    for (let i = 0; i < missing.length; i += PREDICT_CHUNK) {
      if (shouldCancel?.()) break;
      const chunk = missing.slice(i, i + PREDICT_CHUNK);
      const response = await batchPredictCode(
        chunk.map((m) => m.object),
        5,
        version,
      );
      if (!response.success || !response.data) {
        throw new Error(response.error || t.predict.failed);
      }
      map = appendSessions(
        toEntries(
          response.data.results,
          chunk.map((m) => m.index),
        ),
        map,
      );
      done += chunk.length;
      onProgress?.(done, total);
    }
    return map;
  };

  const handleSelectCandidate = (
    objectIndex: number,
    sessionIndex: number,
    candidateIndex: number,
  ) => {
    const sessions = predictionMap[objectIndex];
    if (!sessions || !sessions[sessionIndex]) return;

    const prevSession = sessions[sessionIndex];
    const pairCount = getPairCount(prevSession.prediction);
    const wasUserCard = prevSession.selectedIndex === pairCount;
    const isUserCard = candidateIndex === pairCount;

    const nextMap = { ...predictionMap };
    const updatedSessions = [...sessions];
    updatedSessions[sessionIndex] = { ...updatedSessions[sessionIndex], selectedIndex: candidateIndex };
    nextMap[objectIndex] = updatedSessions;
    setPredictionMap(nextMap);
    saveToDisk(nextMap);

    if (isUserCard && prevSession.userCandidate) {
      onSelectionSync(objectIndex, sessionIndex, { ...prevSession, selectedIndex: candidateIndex }, 'add');
    } else if (wasUserCard && !isUserCard) {
      onSelectionSync(objectIndex, sessionIndex, prevSession, 'remove');
    }
  };

  const handleUserCandidateChange = (
    objectIndex: number,
    sessionIndex: number,
    candidate: UserCandidate,
  ) => {
    const sessions = predictionMap[objectIndex];
    if (!sessions || !sessions[sessionIndex]) return;

    const nextMap = { ...predictionMap };
    const updatedSessions = [...sessions];
    updatedSessions[sessionIndex] = { ...updatedSessions[sessionIndex], userCandidate: candidate };
    nextMap[objectIndex] = updatedSessions;
    setPredictionMap(nextMap);
    saveToDisk(nextMap);

    const pairCount = getPairCount(updatedSessions[sessionIndex].prediction);
    if (updatedSessions[sessionIndex].selectedIndex === pairCount) {
      onSelectionSync(objectIndex, sessionIndex, updatedSessions[sessionIndex], 'add');
    }
  };

  return {
    predictionMap,
    setPredictionMap,
    appendSessions,
    toEntries,
    ensureAllPredicted,
    handleSelectCandidate,
    handleUserCandidateChange,
    toSession,
  };
}
