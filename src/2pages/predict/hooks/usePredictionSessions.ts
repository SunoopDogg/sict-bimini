import { useState } from 'react';

import type {
  PredictionCandidates,
  PredictionResult,
  PredictionSession,
} from '@/src/5entities/prediction';
import { savePredictionsAction } from '@/src/4features/predict-code';

function toSession(candidates: PredictionCandidates): PredictionSession {
  return {
    candidates: candidates.predictions,
    selectedIndex: 0,
    predicted_at: new Date().toISOString(),
  };
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
  const [predictionMap, setPredictionMap] = useState<
    Record<string, PredictionSession[]>
  >({});

  const saveToDisk = (nextMap: Record<string, PredictionSession[]>) => {
    if (selectedFile) {
      savePredictionsAction(selectedFile, nextMap);
    }
  };

  const appendSessions = (
    entries: { index: number; session: PredictionSession }[],
  ) => {
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
    const wasUserCard =
      prevSession.selectedIndex === prevSession.candidates.length;
    const isUserCard = candidateIndex === prevSession.candidates.length;

    const nextMap = { ...predictionMap };
    const updatedSessions = [...sessions];
    updatedSessions[sessionIndex] = {
      ...updatedSessions[sessionIndex],
      selectedIndex: candidateIndex,
    };
    nextMap[objectIndex] = updatedSessions;
    setPredictionMap(nextMap);
    saveToDisk(nextMap);

    // Sync selections file
    if (isUserCard && prevSession.userCandidate) {
      onSelectionSync(
        objectIndex,
        sessionIndex,
        { ...prevSession, selectedIndex: candidateIndex },
        'add',
      );
    } else if (wasUserCard && !isUserCard) {
      onSelectionSync(objectIndex, sessionIndex, prevSession, 'remove');
    }
  };

  const handleUserCandidateChange = (
    objectIndex: number,
    sessionIndex: number,
    candidate: PredictionResult,
  ) => {
    const sessions = predictionMap[objectIndex];
    if (!sessions || !sessions[sessionIndex]) return;

    const nextMap = { ...predictionMap };
    const updatedSessions = [...sessions];
    updatedSessions[sessionIndex] = {
      ...updatedSessions[sessionIndex],
      userCandidate: candidate,
    };
    nextMap[objectIndex] = updatedSessions;
    setPredictionMap(nextMap);
    saveToDisk(nextMap);

    // If user card is currently selected, update the saved selection too
    if (
      updatedSessions[sessionIndex].selectedIndex ===
      updatedSessions[sessionIndex].candidates.length
    ) {
      onSelectionSync(
        objectIndex,
        sessionIndex,
        updatedSessions[sessionIndex],
        'add',
      );
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
