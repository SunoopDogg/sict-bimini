'use client';

import { FileSpreadsheet } from 'lucide-react';

import type { XlsxFileInfo } from '@/5entities/xlsx-file';
import { useLocale } from '@/6shared/i18n';
import { formatDateTime, formatFileSize } from '@/6shared/lib/format';
import { SelectableCardList } from '@/6shared/ui/SelectableCardList';

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
  const { t, locale } = useLocale();

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
      <SelectableCardList
        items={files}
        getKey={(file) => file.name}
        selectedKey={selectedFile}
        onSelect={onSelect}
        renderIcon={() => (
          <FileSpreadsheet className="h-8 w-8 shrink-0 text-green-600" />
        )}
        renderTitle={(file) => file.name}
        renderSubtitle={(file) =>
          `${formatFileSize(file.size)} · ${formatDateTime(file.modifiedAt, locale)}`
        }
      />
    </div>
  );
}
