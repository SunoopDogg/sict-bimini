'use client';

import { useEffect, useState, useTransition } from 'react';

import { BatchPredictionResultWidget } from '@/src/3widgets/batch-prediction-result';
import {
  FileUploadZone,
  FileListSelect,
  ObjectListPanel,
  listXlsxFilesAction,
  readJsonFileAction,
  uploadAndConvertXlsxAction,
  type UploadStatus,
} from '@/src/4features/predict-code';
import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { BatchPredictResult } from '@/src/5entities/prediction';
import { batchPredictCode, predictSingleCode } from '@/src/6shared/api';
import { Alert, AlertDescription } from '@/src/6shared/ui/primitive/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/src/6shared/ui/primitive/card';

export default function PredictPage() {
  const [files, setFiles] = useState<XlsxFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>();
  const [objects, setObjects] = useState<BIMObjectInput[]>([]);
  const [result, setResult] = useState<BatchPredictResult | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const [error, setError] = useState<string>();
  const [isPredicting, startPrediction] = useTransition();
  const [predictingIndex, setPredictingIndex] = useState<number | null>(null);

  const reloadFileList = async () => {
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
      await reloadFileList();
      setSelectedFile(response.data.file?.name);
      setTimeout(() => setUploadStatus('idle'), 2000);
    } else {
      setUploadStatus('idle');
      setError(response.error || '파일 업로드에 실패했습니다.');
    }
  };

  const handlePredict = () => {
    setError(undefined);
    startPrediction(async () => {
      const response = await batchPredictCode(objects);

      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || '예측에 실패했습니다.');
      }
    });
  };

  const handleSinglePredict = (obj: BIMObjectInput, index: number) => {
    setError(undefined);
    setPredictingIndex(index);
    startPrediction(async () => {
      const response = await predictSingleCode(obj);

      if (response.success && response.data) {
        setResult({
          results: [{ input: obj, prediction: response.data, error: null }],
          total: 1,
          successful: 1,
          failed: 0,
        });
      } else {
        setError(response.error || '예측에 실패했습니다.');
      }
      setPredictingIndex(null);
    });
  };

  useEffect(() => {
    const loadInitialFiles = async () => {
      await reloadFileList();
    };
    loadInitialFiles();
  }, []);

  useEffect(() => {
    if (!selectedFile) return;

    const loadObjects = async () => {
      setResult(null);
      setError(undefined);
      setIsLoadingObjects(true);

      const response = await readJsonFileAction(selectedFile);

      if (response.success && response.data) {
        setObjects(response.data);
      } else {
        setObjects([]);
        setError(response.error || '객체 데이터를 불러올 수 없습니다.');
      }

      setIsLoadingObjects(false);
    };

    loadObjects();
  }, [selectedFile]);

  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="mb-8 text-center text-3xl font-bold">
        KBIMS 부위코드 예측
      </h1>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-[280px_1.2fr_1fr] gap-4">
        {/* Panel 1: File List */}
        <div className="min-h-0">
          <Card className="flex h-full flex-col">
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
            onPredict={handlePredict}
            onRowClick={handleSinglePredict}
          />
        </div>

        {/* Panel 3: Prediction Results */}
        <div className="min-h-0">
          <BatchPredictionResultWidget result={result} />
        </div>
      </div>
    </main>
  );
}
