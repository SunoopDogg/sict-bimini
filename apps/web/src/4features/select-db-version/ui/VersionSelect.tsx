'use client';

import { Database, Loader2 } from 'lucide-react';

import type { DbVersion } from '@/5entities/db-version';
import { useLocale } from '@/6shared/i18n';
import { DbVersionViewButton } from '@/6shared/ui/DbVersionViewButton';
import { SelectableCardList } from '@/6shared/ui/SelectableCardList';

interface VersionSelectProps {
  versions: DbVersion[];
  value?: string;
  onChange: (version: string) => void;
  isLoading?: boolean;
  error?: string | null;
  /** When provided, each version shows an eye button that views its contents. */
  onView?: (version: string) => void;
}

export function VersionSelect({
  versions,
  value,
  onChange,
  isLoading,
  error,
  onView,
}: VersionSelectProps) {
  const { t } = useLocale();

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
        {t.version.none}
      </div>
    );
  }

  return (
    <SelectableCardList
      items={versions}
      getKey={(v) => v.name}
      selectedKey={value}
      onSelect={onChange}
      renderIcon={() => (
        <Database className="h-7 w-7 shrink-0 text-blue-600" />
      )}
      renderTitle={(v) => v.name}
      renderSubtitle={(v) => t.version.items(v.points)}
      renderAction={
        onView
          ? (v) => <DbVersionViewButton version={v.name} onView={onView} />
          : undefined
      }
    />
  );
}
