'use client';

import { useEffect, useState, useTransition } from 'react';

import { usePredictionSessions } from '@/src/2pages/predict/hooks/usePredictionSessions';
import { useUserSelections } from '@/src/2pages/predict/hooks/useUserSelections';
import { ObjectPredictionPanel } from '@/src/3widgets/object-prediction-panel';
import { UserSelectionPanel } from '@/src/3widgets/user-selection-panel';
import { BimAttributeTableModal } from '@/src/4features/bim-attribute-viewer';
import { ServerStatusBadge } from '@/src/4features/server-status';
import {
  FileListSelect,
  FileUploadZone,
  ObjectListPanel,
  type UploadStatus,
  listXlsxFilesAction,
  readJsonFileAction,
  uploadAndConvertXlsxAction,
  loadUserSelectionsAction,
} from '@/src/4features/manage-file';
import { loadPredictionsAction } from '@/src/4features/predict-code';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { PredictionSession } from '@/src/5entities/prediction';
import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';
import { batchPredictCode, predictSingleCode } from '@/src/6shared/api';
import { Alert, AlertDescription } from '@/src/6shared/ui/primitive/alert';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';

type DataSource =
  | { type: 'xlsx'; fileName: string }
  | { type: 'selection'; fileName: string }
  | null;

export default function PredictPage() {
  const [files, setFiles] = useState<XlsxFileInfo[]>([]);
  const [activeSource, setDataSource] = useState<DataSource>(null);
  const [objects, setObjects] = useState<BIMObjectInput[]>([]);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState<number | null>(
    null,
  );
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const [error, setError] = useState<string>();
  const [isPredicting, startPrediction] = useTransition();
  const [predictingIndex, setPredictingIndex] = useState<number | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );

  const selectedFile =
    activeSource?.type === 'xlsx' ? activeSource.fileName : undefined;
  const selectedSelectionFile =
    activeSource?.type === 'selection' ? activeSource.fileName : undefined;

  const {
    selectionFiles,
    refreshSelectionFiles,
    addToSelections,
    removeFromSelections,
    loadInitialSelections,
    setSelectionsFromData,
    syncSelectionsFromMap,
  } = useUserSelections(objects);

  const {
    predictionMap,
    setPredictionMap,
    appendSessions,
    handleSelectCandidate,
    handleUserCandidateChange,
    toSession,
  } = usePredictionSessions({
    selectedFile,
    onSelectionSync: (objectIndex, sessionIndex, session, action) => {
      if (action === 'add') {
        addToSelections(objectIndex, sessionIndex, session);
      } else {
        removeFromSelections(objectIndex);
      }
    },
  });

  const refreshFileList = async () => {
    const response = await listXlsxFilesAction();
    if (response.success && response.data) {
      setFiles(response.data);
    } else {
      setError(response.error || '파일 목록을 불러올 수 없습니다.');
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploadStatus('uploading');
    setError(undefined);

    const formData = new FormData();
    formData.append('file', file);

    const response = await uploadAndConvertXlsxAction(formData, true);

    if (response.success && response.data) {
      setUploadStatus('done');
      await refreshFileList();
      if (response.data.file?.name) {
        setDataSource({ type: 'xlsx', fileName: response.data.file.name });
      }
      setTimeout(() => setUploadStatus('idle'), 2000);
    } else {
      setUploadStatus('idle');
      setError(response.error || '파일 업로드에 실패했습니다.');
    }
  };

  const handleSelectXlsxFile = (fileName: string) => {
    setDataSource({ type: 'xlsx', fileName });
  };

  const handleSelectSelectionFile = async (fileName: string) => {
    setDataSource({ type: 'selection', fileName });
  };

  const handleBatchPredict = () => {
    setError(undefined);
    const selectedObjects = objects.filter((_, i) => selectedIndices.has(i));
    const selectedIndicesArray = Array.from(selectedIndices);
    startPrediction(async () => {
      const response = await batchPredictCode(selectedObjects);

      if (response.success && response.data) {
        const entries = response.data.results
          .map((item, i) =>
            item.prediction
              ? {
                  index: selectedIndicesArray[i],
                  session: toSession(item.prediction),
                }
              : null,
          )
          .filter(
            (e): e is { index: number; session: PredictionSession } =>
              e !== null,
          );
        appendSessions(entries);
        setSelectedIndices(new Set());
      } else {
        setError(response.error || '예측에 실패했습니다.');
      }
    });
  };

  const handleSinglePredict = (index: number) => {
    setError(undefined);
    setPredictingIndex(index);
    startPrediction(async () => {
      const response = await predictSingleCode(objects[index]);

      if (response.success && response.data) {
        appendSessions([{ index, session: toSession(response.data) }]);
      } else {
        setError(response.error || '예측에 실패했습니다.');
      }
      setPredictingIndex(null);
    });
  };

  // Initial load
  useEffect(() => {
    const loadInitial = async () => {
      await Promise.all([refreshFileList(), refreshSelectionFiles()]);
      await loadInitialSelections();
    };
    loadInitial();
  }, []);

  // Load objects when xlsx file is selected
  useEffect(() => {
    if (activeSource?.type !== 'xlsx') return;
    const fileName = activeSource.fileName;

    const loadObjects = async () => {
      setSelectedObjectIndex(null);
      setError(undefined);
      setSelectedIndices(new Set());
      setIsLoadingObjects(true);

      let loadedObjects: BIMObjectInput[] = [];
      const response = await readJsonFileAction(fileName);

      if (response.success && response.data) {
        loadedObjects = response.data;
        setObjects(response.data);
      } else {
        setObjects([]);
        setError(response.error || '객체 데이터를 불러올 수 없습니다.');
      }

      let loadedMap: Record<string, PredictionSession[]> = {};
      const predResult = await loadPredictionsAction(fileName);
      if (predResult.success && predResult.data) {
        loadedMap = predResult.data;
        setPredictionMap(loadedMap);
      } else {
        setPredictionMap({});
      }

      syncSelectionsFromMap(loadedMap, loadedObjects);
      setIsLoadingObjects(false);
    };

    loadObjects();
  }, [activeSource?.type === 'xlsx' ? activeSource.fileName : null]);

  // Load selections when selection file is selected
  useEffect(() => {
    if (activeSource?.type !== 'selection') return;

    const loadSelections = async () => {
      setSelectedObjectIndex(null);
      setError(undefined);
      setSelectedIndices(new Set());
      setIsLoadingObjects(true);

      const response = await loadUserSelectionsAction();

      if (response.success && response.data) {
        setSelectionsFromData(response.data);
        const selObjects = response.data.map((sel) => sel.object);
        setObjects(selObjects);
        const map: Record<string, PredictionSession[]> = {};
        for (let i = 0; i < response.data.length; i++) {
          const sel = response.data[i];
          map[i] = [
            {
              candidates: [],
              userCandidate: sel.candidate,
              selectedIndex: 0,
              predicted_at: sel.selectedAt,
            },
          ];
        }
        setPredictionMap(map);
      } else {
        setSelectionsFromData([]);
        setObjects([]);
        setPredictionMap({});
        setError(response.error || '사용자 선택을 불러올 수 없습니다.');
      }

      setIsLoadingObjects(false);
    };

    loadSelections();
  }, [activeSource?.type === 'selection' ? activeSource.fileName : null]);

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="relative mb-8 flex items-center justify-center">
        <div className="absolute left-0">
          <BimAttributeTableModal />
        </div>
        <h1 className="text-3xl font-bold">KBIMS 코드 예측</h1>
        <div className="absolute right-0">
          <ServerStatusBadge />
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-[280px_1.2fr_1fr] gap-4">
        {/* Panel 1: File List + User Selections */}
        <div className="flex flex-col gap-4 min-h-0">
          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>파일</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col gap-4 overflow-hidden">
              <FileUploadZone
                onUpload={handleFileUpload}
                uploadStatus={uploadStatus}
              />
              <div className="flex-1 overflow-y-auto">
                <FileListSelect
                  files={files}
                  selectedFile={selectedFile}
                  onSelect={handleSelectXlsxFile}
                />
              </div>
            </CardContent>
          </Card>
          <UserSelectionPanel
            files={selectionFiles}
            selectedFile={selectedSelectionFile}
            onSelect={handleSelectSelectionFile}
          />
        </div>

        {/* Panel 2: Object List */}
        <div className="min-h-0">
          <ObjectListPanel
            selectedFile={activeSource?.fileName}
            objects={objects}
            isLoading={isLoadingObjects}
            isPredicting={isPredicting}
            predictingIndex={predictingIndex}
            selectedIndices={selectedIndices}
            onSelectionChange={setSelectedIndices}
            predictionMap={predictionMap}
            focusedIndex={selectedObjectIndex}
            onPredict={handleBatchPredict}
            onRowClick={(_obj: BIMObjectInput, index: number) => {
              setSelectedObjectIndex(index);
            }}
          />
        </div>

        {/* Panel 3: Prediction Results */}
        <div className="min-h-0">
          <ObjectPredictionPanel
            object={
              selectedObjectIndex !== null
                ? (objects[selectedObjectIndex] ?? null)
                : null
            }
            sessions={
              selectedObjectIndex !== null
                ? (predictionMap[selectedObjectIndex] ?? [])
                : []
            }
            isPredicting={
              predictingIndex === selectedObjectIndex || isPredicting
            }
            onPredict={() => {
              if (selectedObjectIndex !== null) {
                handleSinglePredict(selectedObjectIndex);
              }
            }}
            onSelectCandidate={(sessionIndex, candidateIndex) => {
              if (selectedObjectIndex !== null) {
                handleSelectCandidate(
                  selectedObjectIndex,
                  sessionIndex,
                  candidateIndex,
                );
              }
            }}
            onUserCandidateChange={(sessionIndex, candidate) => {
              if (selectedObjectIndex !== null) {
                handleUserCandidateChange(
                  selectedObjectIndex,
                  sessionIndex,
                  candidate,
                );
              }
            }}
          />
        </div>
      </div>
    </main>
  );
}
