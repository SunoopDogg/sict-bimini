'use client';

import { Check, FileText } from 'lucide-react';

import type { SelectionFileInfo } from '@/5entities/prediction';
import { useLocale } from '@/6shared/i18n';
import { cn } from '@/6shared/lib/cn';
import { formatDateTime } from '@/6shared/lib/format';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/6shared/ui/primitive/card';

interface UserSelectionPanelProps {
  files: SelectionFileInfo[];
  selectedFile?: string;
  onSelect: (fileName: string) => void;
}

export function UserSelectionPanel({
  files,
  selectedFile,
  onSelect,
}: UserSelectionPanelProps) {
  const { t } = useLocale();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{t.userSel.title}</CardTitle>
      </CardHeader>
      <CardContent>
        {files.length === 0 ? (
          <p className="text-muted-foreground text-center text-xs py-4">
            {t.userSel.noFiles}
          </p>
        ) : (
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
                  <FileText className="h-8 w-8 shrink-0 text-blue-600" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium text-sm">{file.name}</p>
                    <p className="text-muted-foreground text-xs">
                      {t.userSel.items(file.itemCount)} · {formatDateTime(file.modifiedAt)}
                    </p>
                  </div>
                  {selectedFile === file.name && (
                    <Check className="text-primary h-5 w-5 shrink-0" />
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
