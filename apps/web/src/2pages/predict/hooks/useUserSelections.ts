import { useRef, useState } from 'react';

import type { BIMObject } from '@/5entities/bim-object';
import { EMPTY_BIM_OBJECT } from '@/5entities/bim-object';
import type { PredictionSession, SelectionFileInfo, UserSelection } from '@/5entities/prediction';
import { getPairCount } from '@/5entities/prediction';
import {
  saveUserSelectionsAction,
  loadUserSelectionsAction,
  listSelectionFilesAction,
} from '@/4features/manage-file';

function buildUserSelection(
  objectIndex: number,
  sessionIndex: number,
  session: PredictionSession,
  objects: BIMObject[],
): UserSelection | null {
  if (!session.userCandidate) return null;
  const obj = objects[objectIndex];
  const pairCount = getPairCount(session.prediction);
  const kbimsConf = session.selectedIndex < pairCount
    ? (session.prediction.kbims.candidates[session.selectedIndex]?.llm_confidence ?? 0)
    : 0;
  const ppsConf = session.selectedIndex < pairCount
    ? (session.prediction.pps.candidates[session.selectedIndex]?.llm_confidence ?? 0)
    : 0;
  return {
    objectIndex,
    objectName: obj?.name,
    sessionIndex,
    kbims_code: session.userCandidate.kbims_code,
    pps_code: session.userCandidate.pps_code,
    kbims_confidence: kbimsConf,
    pps_confidence: ppsConf,
    object: obj ?? EMPTY_BIM_OBJECT,
    selectedAt: new Date().toISOString(),
  };
}

export function useUserSelections(objects: BIMObject[]) {
  const [selectionFiles, setSelectionFiles] = useState<SelectionFileInfo[]>([]);
  const savedSelectionsRef = useRef<UserSelection[]>([]);

  const refreshSelectionFiles = async () => {
    const response = await listSelectionFilesAction();
    if (response.success && response.data) {
      setSelectionFiles(response.data);
    }
  };

  const updateSelections = (next: UserSelection[]) => {
    savedSelectionsRef.current = next;
    saveUserSelectionsAction(next).then(refreshSelectionFiles);
  };

  const addToSelections = (
    objectIndex: number,
    sessionIndex: number,
    session: PredictionSession,
  ) => {
    const newSel = buildUserSelection(objectIndex, sessionIndex, session, objects);
    if (!newSel) return;
    const next = [
      ...savedSelectionsRef.current.filter((s) => s.objectName !== newSel.objectName),
      newSel,
    ];
    updateSelections(next);
  };

  const removeFromSelections = (objectIndex: number) => {
    const objectName = objects[objectIndex]?.name;
    const next = savedSelectionsRef.current.filter((s) => s.objectName !== objectName);
    updateSelections(next);
  };

  const loadInitialSelections = async () => {
    const selResult = await loadUserSelectionsAction();
    if (selResult.success && selResult.data) {
      savedSelectionsRef.current = selResult.data;
    }
  };

  const setSelectionsFromData = (data: UserSelection[]) => {
    savedSelectionsRef.current = data;
  };

  const syncSelectionsFromMap = (
    loadedMap: Record<string, PredictionSession[]>,
    loadedObjects: BIMObject[],
  ) => {
    const newSelections: UserSelection[] = [];
    for (const [key, sessions] of Object.entries(loadedMap)) {
      const objectIndex = Number(key);
      for (let sessionIndex = 0; sessionIndex < sessions.length; sessionIndex++) {
        const session = sessions[sessionIndex];
        if (!session.prediction) continue; // skip old-format sessions from disk
        const pairCount = getPairCount(session.prediction);
        if (session.selectedIndex === pairCount && session.userCandidate) {
          const sel = buildUserSelection(objectIndex, sessionIndex, session, loadedObjects);
          if (sel) newSelections.push(sel);
        }
      }
    }
    if (newSelections.length > 0) {
      const others = savedSelectionsRef.current.filter(
        (s) => !newSelections.some((ns) => ns.objectName === s.objectName),
      );
      updateSelections([...others, ...newSelections]);
    }
  };

  return {
    selectionFiles,
    savedSelectionsRef,
    refreshSelectionFiles,
    addToSelections,
    removeFromSelections,
    loadInitialSelections,
    setSelectionsFromData,
    syncSelectionsFromMap,
  };
}
