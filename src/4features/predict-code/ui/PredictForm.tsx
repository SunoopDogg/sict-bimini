'use client';

import { useEffect, useState, useTransition } from 'react';

import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';
import { Alert, AlertDescription } from '@/src/6shared/ui/primitive/alert';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';

import {
  listXlsxFilesAction,
  uploadAndConvertXlsxAction,
} from '../api/actions';
import type { BatchPredictionResult, UploadStatus } from '../api/types';
import { FileListSelect } from './FileListSelect';
import { FileUploadZone } from './FileUploadZone';

interface PredictFormProps {
  onResult?: (state: BatchPredictionResult) => void;
}

export function PredictForm({ onResult }: PredictFormProps) {
  const [files, setFiles] = useState<XlsxFileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>();
  const [error, setError] = useState<string>();
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [isPredicting, startPrediction] = useTransition();

  const reloadFileList = async () => {
    const result = await listXlsxFilesAction();
    if (result.success && result.files) {
      setFiles(result.files);
    }
  };

  useEffect(() => {
    void listXlsxFilesAction().then((result) => {
      if (result.success && result.files) {
        setFiles(result.files);
      }
    });
  }, []);

  const handleFileUpload = async (file: File) => {
    setError(undefined);
    setUploadStatus('uploading');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const result = await uploadAndConvertXlsxAction(formData);

      if (result.success && result.file) {
        setUploadStatus('done');
        await reloadFileList();
        setSelectedFile(result.file.name);
      } else {
        setError(result.error);
        setUploadStatus('idle');
      }
    } catch {
      setError('파일 업로드 중 오류가 발생했습니다.');
      setUploadStatus('idle');
    }
  };

  const handleFileSelect = (fileName: string) => {
    setSelectedFile(fileName);
    if (uploadStatus === 'done') {
      setUploadStatus('idle');
    }
  };

  const handlePredictionSubmit = () => {};

  return (
    <Card>
      <CardHeader>
        <CardTitle>xlsx 파일 업로드</CardTitle>
        <CardDescription>
          BIM 객체 데이터가 포함된 xlsx 파일을 업로드하면 KBIMS 부위코드를 일괄
          예측합니다.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <FileUploadZone
          onUpload={handleFileUpload}
          uploadStatus={uploadStatus}
        />

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <FileListSelect
          files={files}
          selectedFile={selectedFile}
          onSelect={handleFileSelect}
        />

        <Button
          onClick={handlePredictionSubmit}
          disabled={!selectedFile || isPredicting}
          className="w-full"
        >
          {isPredicting ? '예측 중...' : '부위코드 일괄 예측'}
        </Button>
      </CardContent>
    </Card>
  );
}
