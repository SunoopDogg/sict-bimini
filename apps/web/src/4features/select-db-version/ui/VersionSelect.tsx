'use client';

import type { DbVersion } from '@/5entities/db-version';

interface VersionSelectProps {
  versions: DbVersion[];
  value?: string;
  onChange: (version: string) => void;
  disabled?: boolean;
}

export function VersionSelect({
  versions,
  value,
  onChange,
  disabled,
}: VersionSelectProps) {
  return (
    <select
      className="border-input bg-background h-9 rounded-md border px-3 text-sm"
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled || versions.length === 0}
      aria-label="DB version"
    >
      {versions.length === 0 && <option value="">버전 없음</option>}
      {versions.map((v) => (
        <option key={v.name} value={v.name}>
          {v.name} ({v.points})
        </option>
      ))}
    </select>
  );
}
