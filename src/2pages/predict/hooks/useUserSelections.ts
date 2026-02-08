import { useRef, useState } from 'react';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import { EMPTY_BIM_OBJECT } from '@/src/5entities/bim-object';
import type { PredictionSession, UserSelection } from '@/src/5entities/prediction';
import {
  saveUserSelectionsAction,
  loadUserSelectionsAction,
  listSelectionFilesAction,
} from '@/src/4features/manage-file';
import type { SelectionFileInfo } from '@/src/5entities/prediction';

export function useUserSelections(objects: BIMObjectInput[]) {
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
    if (!session.userCandidate) return;
    const obj = objects[objectIndex];
    const newSel: UserSelection = {
      objectIndex,
      objectName: obj?.name ?? '',
      sessionIndex,
      candidate: session.userCandidate,
      object: obj ?? EMPTY_BIM_OBJECT,
      selectedAt: new Date().toISOString(),
    };
    const next = [
      ...savedSelectionsRef.current.filter(
        (s) => s.objectName !== newSel.objectName,
      ),
      newSel,
    ];
    updateSelections(next);
  };

  const removeFromSelections = (objectIndex: number) => {
    const objectName = objects[objectIndex]?.name ?? '';
    const next = savedSelectionsRef.current.filter(
      (s) => s.objectName !== objectName,
    );
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
    loadedObjects: BIMObjectInput[],
  ) => {
    const newSelections: UserSelection[] = [];
    for (const [key, sessions] of Object.entries(loadedMap)) {
      const objectIndex = Number(key);
      for (
        let sessionIndex = 0;
        sessionIndex < sessions.length;
        sessionIndex++
      ) {
        const session = sessions[sessionIndex];
        if (
          session.selectedIndex === session.candidates.length &&
          session.userCandidate
        ) {
          newSelections.push({
            objectIndex,
            objectName: loadedObjects[objectIndex]?.name ?? '',
            sessionIndex,
            candidate: session.userCandidate,
            object: loadedObjects[objectIndex] ?? EMPTY_BIM_OBJECT,
            selectedAt: new Date().toISOString(),
          });
        }
      }
    }

    if (newSelections.length > 0) {
      const otherSelections = savedSelectionsRef.current.filter(
        (s) => !newSelections.some((ns) => ns.objectName === s.objectName),
      );
      const merged = [...otherSelections, ...newSelections];
      updateSelections(merged);
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
