'use client';

import { Check, Database, Loader2 } from 'lucide-react';

import type { DbVersion } from '@/5entities/db-version';
import { cn } from '@/6shared/lib/cn';

interface VersionSelectProps {
  versions: DbVersion[];
  value?: string;
  onChange: (version: string) => void;
  isLoading?: boolean;
  error?: string | null;
}

export function VersionSelect({
  versions,
  value,
  onChange,
  isLoading,
  error,
}: VersionSelectProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="text-muted-foreground h-5 w-5 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-destructive py-6 text-center text-sm">{error}</div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="text-muted-foreground py-6 text-center text-sm">
        버전 없음
      </div>
    );
  }

  return (
    <ul className="space-y-2">
      {versions.map((version) => (
        <li key={version.name}>
          <button
            type="button"
            onClick={() => onChange(version.name)}
            className={cn(
              'hover:bg-accent flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors',
              value === version.name
                ? 'border-primary bg-primary/5'
                : 'border-border',
            )}
          >
            <Database className="h-7 w-7 shrink-0 text-blue-600" />
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{version.name}</p>
              <p className="text-muted-foreground text-xs">
                {version.points.toLocaleString()}개
              </p>
            </div>
            {value === version.name && (
              <Check className="text-primary h-5 w-5 shrink-0" />
            )}
          </button>
        </li>
      ))}
    </ul>
  );
}
