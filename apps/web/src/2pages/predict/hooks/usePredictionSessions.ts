import { useState } from 'react';

import type { CombinedPredictionResponse, PredictionSession } from '@/5entities/prediction';
import { savePredictionsAction } from '@/4features/predict-code';

function toSession(prediction: CombinedPredictionResponse): PredictionSession {
  return {
    prediction,
    selectedIndex: 0,
    predicted_at: new Date().toISOString(),
  };
}

function getPairCount(session: PredictionSession): number {
  return Math.min(
    session.prediction.kbims.candidates.length,
    session.prediction.pps.candidates.length,
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
  const [predictionMap, setPredictionMap] = useState<Record<string, PredictionSession[]>>({});

  const saveToDisk = (nextMap: Record<string, PredictionSession[]>) => {
    if (selectedFile) {
      savePredictionsAction(selectedFile, nextMap);
    }
  };

  const appendSessions = (entries: { index: number; session: PredictionSession }[]) => {
    const nextMap = { ...predictionMap };
    for (const { index, session } of entries) {
      const existing = nextMap[index] ?? [];
      nextMap[index] = [...existing, session];
    }
    setPredictionMap(nextMap);
    saveToDisk(nextMap);
  };

  const handleSelectCandidate = (
    objectIndex: number,
    sessionIndex: number,
    candidateIndex: number,
  ) => {
    const sessions = predictionMap[objectIndex];
    if (!sessions || !sessions[sessionIndex]) return;

    const prevSession = sessions[sessionIndex];
    const pairCount = getPairCount(prevSession);
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
    candidate: { kbims_code: string; pps_code: string; reasoning?: string },
  ) => {
    const sessions = predictionMap[objectIndex];
    if (!sessions || !sessions[sessionIndex]) return;

    const nextMap = { ...predictionMap };
    const updatedSessions = [...sessions];
    updatedSessions[sessionIndex] = { ...updatedSessions[sessionIndex], userCandidate: candidate };
    nextMap[objectIndex] = updatedSessions;
    setPredictionMap(nextMap);
    saveToDisk(nextMap);

    const pairCount = getPairCount(updatedSessions[sessionIndex]);
    if (updatedSessions[sessionIndex].selectedIndex === pairCount) {
      onSelectionSync(objectIndex, sessionIndex, updatedSessions[sessionIndex], 'add');
    }
  };

  return {
    predictionMap,
    setPredictionMap,
    appendSessions,
    handleSelectCandidate,
    handleUserCandidateChange,
    toSession,
  };
}
