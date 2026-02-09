'use client';

import { useEffect, useState, useTransition } from 'react';

import { ObjectPredictionPanel } from '@/src/3widgets/object-prediction-panel';
import { ServerStatusBadge } from '@/src/4features/server-status';
import {
  FileListSelect,
  FileUploadZone,
  ObjectListPanel,
  type UploadStatus,
  listXlsxFilesAction,
  readJsonFileAction,
  uploadAndConvertXlsxAction,
} from '@/src/4features/manage-file';
import {
  loadPredictionsAction,
  savePredictionsAction,
} from '@/src/4features/predict-code';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { PredictionResult } from '@/src/5entities/prediction';
import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';
import { batchPredictCode, predictSingleCode } from '@/src/6shared/api';
import { Alert, AlertDescription } from '@/src/6shared/ui/primitive/alert';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';

function predictionMapToRecord(
  map: Map<number, PredictionResult[]>,
): Record<string, PredictionResult[]> {
  const record: Record<string, PredictionResult[]> = {};
  for (const [key, value] of map) {
    record[String(key)] = value;
  }
  return record;
}

export default function PredictPage() {
  const [files, setFiles] = useState<XlsxFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>();
  const [objects, setObjects] = useState<BIMObjectInput[]>([]);
  const [predictionMap, setPredictionMap] = useState<
    Map<number, PredictionResult[]>
  >(new Map());
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

  const appendPredictions = (
    entries: { index: number; prediction: PredictionResult }[],
  ) => {
    const now = new Date().toISOString();
    const nextMap = new Map(predictionMap);
    for (const { index, prediction } of entries) {
      const existing = nextMap.get(index) ?? [];
      nextMap.set(index, [...existing, { ...prediction, predicted_at: now }]);
    }
    setPredictionMap(nextMap);
    if (selectedFile) {
      savePredictionsAction(selectedFile, predictionMapToRecord(nextMap));
    }
  };

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
      setSelectedFile(response.data.file?.name);
      setTimeout(() => setUploadStatus('idle'), 2000);
    } else {
      setUploadStatus('idle');
      setError(response.error || '파일 업로드에 실패했습니다.');
    }
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
              ? { index: selectedIndicesArray[i], prediction: item.prediction }
              : null,
          )
          .filter(
            (e): e is { index: number; prediction: PredictionResult } =>
              e !== null,
          );
        appendPredictions(entries);
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
        appendPredictions([{ index, prediction: response.data }]);
      } else {
        setError(response.error || '예측에 실패했습니다.');
      }
      setPredictingIndex(null);
    });
  };

  useEffect(() => {
    const loadInitialFiles = async () => {
      await refreshFileList();
    };
    loadInitialFiles();
  }, []);

  useEffect(() => {
    if (!selectedFile) return;

    const loadObjects = async () => {
      setSelectedObjectIndex(null);
      setError(undefined);
      setSelectedIndices(new Set());
      setIsLoadingObjects(true);

      const response = await readJsonFileAction(selectedFile);

      if (response.success && response.data) {
        setObjects(response.data);
      } else {
        setObjects([]);
        setError(response.error || '객체 데이터를 불러올 수 없습니다.');
      }

      const predResult = await loadPredictionsAction(selectedFile);
      if (predResult.success && predResult.data) {
        const map = new Map<number, PredictionResult[]>();
        for (const [key, value] of Object.entries(predResult.data)) {
          map.set(Number(key), value);
        }
        setPredictionMap(map);
      } else {
        setPredictionMap(new Map());
      }

      setIsLoadingObjects(false);
    };

    loadObjects();
  }, [selectedFile]);

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="relative mb-8 flex items-center justify-center">
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
        {/* Panel 1: File List */}
        <div className="min-h-0">
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
                  onSelect={setSelectedFile}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Panel 2: Object List */}
        <div className="min-h-0">
          <ObjectListPanel
            selectedFile={selectedFile}
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
            predictions={
              selectedObjectIndex !== null
                ? (predictionMap.get(selectedObjectIndex) ?? [])
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
          />
        </div>
      </div>
    </main>
  );
}
