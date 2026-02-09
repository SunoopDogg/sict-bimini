'use client';

import { CheckCircle, Loader2, Upload } from 'lucide-react';

import { useCallback, useState } from 'react';

import { cn } from '@/src/6shared/lib/cn';

import type { UploadStatus } from '../api/types';

interface FileUploadZoneProps {
  onUpload: (file: File) => Promise<void>;
  uploadStatus?: UploadStatus;
}

export function FileUploadZone({
  onUpload,
  uploadStatus = 'idle',
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const isProcessing = uploadStatus === 'uploading';

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.xlsx')) {
        await onUpload(file);
      }
    },
    [onUpload],
  );

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        await onUpload(file);
        e.target.value = '';
      }
    },
    [onUpload],
  );

  const renderContent = () => {
    switch (uploadStatus) {
      case 'uploading':
        return (
          <>
            <Loader2 className="text-primary h-10 w-10 animate-spin" />
            <div className="text-center">
              <p className="text-sm font-medium">업로드 및 변환 중...</p>
              <p className="text-muted-foreground text-xs">잠시만 기다려주세요</p>
            </div>
          </>
        );
      case 'done':
        return (
          <>
            <CheckCircle className="h-10 w-10 text-green-500" />
            <p className="text-sm font-medium text-green-600">변환 완료!</p>
          </>
        );
      default:
        return (
          <>
            <Upload className="text-muted-foreground h-10 w-10" />
            <div className="text-center">
              <p className="text-sm font-medium">xlsx 파일을 드래그하거나</p>
              <label className="text-primary cursor-pointer text-sm hover:underline">
                파일 선택
                <input
                  type="file"
                  accept=".xlsx"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>
            <p className="text-muted-foreground text-xs">.xlsx 형식만 지원됩니다</p>
          </>
        );
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        'flex flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed p-8 transition-colors',
        isDragging
          ? 'border-primary bg-primary/5'
          : 'border-muted-foreground/25 hover:border-primary/50',
        isProcessing && 'pointer-events-none',
        uploadStatus === 'done' && 'border-green-500 bg-green-50 dark:bg-green-950/20',
      )}
    >
      {renderContent()}
    </div>
  );
}
