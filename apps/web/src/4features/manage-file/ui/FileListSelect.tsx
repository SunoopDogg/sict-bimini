'use client';

import { Check, FileSpreadsheet } from 'lucide-react';

import type { XlsxFileInfo } from '@/5entities/xlsx-file';
import { useLocale } from '@/6shared/i18n';
import { cn } from '@/6shared/lib/cn';
import { formatDateTime, formatFileSize } from '@/6shared/lib/format';

interface FileListSelectProps {
  files: XlsxFileInfo[];
  selectedFile?: string;
  onSelect: (fileName: string) => void;
}

export function FileListSelect({
  files,
  selectedFile,
  onSelect,
}: FileListSelectProps) {
  const { t } = useLocale();

  if (files.length === 0) {
    return (
      <div className="text-muted-foreground py-8 text-center text-sm">
        {t.file.noFiles}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-muted-foreground text-sm font-medium">
        {t.file.uploadedFiles(files.length)}
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
