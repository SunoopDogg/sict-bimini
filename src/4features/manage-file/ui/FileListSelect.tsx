'use client';

import { Check, FileSpreadsheet } from 'lucide-react';

import type { XlsxFileInfo } from '@/src/5entities/xlsx-file';
import { cn } from '@/src/6shared/lib/cn';

interface FileListSelectProps {
  files: XlsxFileInfo[];
  selectedFile?: string;
  onSelect: (fileName: string) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const koreanDateFormatter = new Intl.DateTimeFormat('ko-KR', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
});

function formatDateTime(date: string): string {
  return koreanDateFormatter.format(new Date(date));
}

export function FileListSelect({
  files,
  selectedFile,
  onSelect,
}: FileListSelectProps) {
  if (files.length === 0) {
    return (
      <div className="text-muted-foreground py-8 text-center text-sm">
        업로드된 파일이 없습니다.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-muted-foreground text-sm font-medium">
        업로드된 파일 ({files.length}개)
      </p>
      <ul className="space-y-2">
        {files.map((file) => (
          <li key={file.name}>
            <button
              type="button"
              onClick={() => onSelect(file.name)}
              className={cn(
                'hover:bg-accent flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors',
                selectedFile === file.name
                  ? 'border-primary bg-primary/5'
                  : 'border-border',
              )}
            >
              <FileSpreadsheet className="h-8 w-8 shrink-0 text-green-600" />
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{file.name}</p>
                <p className="text-muted-foreground text-xs">
                  {formatFileSize(file.size)} ·{' '}
                  {formatDateTime(file.modifiedAt)}
                </p>
              </div>
              {selectedFile === file.name && (
                <Check className="text-primary h-5 w-5 shrink-0" />
              )}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
