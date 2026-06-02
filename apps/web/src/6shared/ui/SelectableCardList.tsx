'use client';

import { Check } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '@/6shared/lib/cn';

interface SelectableCardListProps<T> {
  items: T[];
  getKey: (item: T) => string;
  selectedKey?: string;
  onSelect: (key: string) => void;
  renderIcon: (item: T) => ReactNode;
  renderTitle: (item: T) => ReactNode;
  renderSubtitle?: (item: T) => ReactNode;
}

/**
 * A vertical list of selectable cards: icon + title/subtitle, active highlight,
 * and a trailing check on the selected item. Shared by FileListSelect and
 * VersionSelect; callers own empty/loading/error states and any list header.
 */
export function SelectableCardList<T>({
  items,
  getKey,
  selectedKey,
  onSelect,
  renderIcon,
  renderTitle,
  renderSubtitle,
}: SelectableCardListProps<T>) {
  return (
    <ul className="space-y-2">
      {items.map((item) => {
        const key = getKey(item);
        const selected = selectedKey === key;
        return (
          <li key={key}>
            <button
              type="button"
              onClick={() => onSelect(key)}
              className={cn(
                'hover:bg-accent flex w-full items-center gap-3 rounded-lg border p-3 text-left transition-colors',
                selected ? 'border-primary bg-primary/5' : 'border-border',
              )}
            >
              {renderIcon(item)}
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">{renderTitle(item)}</p>
                {renderSubtitle && (
                  <p className="text-muted-foreground text-xs">
                    {renderSubtitle(item)}
                  </p>
                )}
              </div>
              {selected && <Check className="text-primary h-5 w-5 shrink-0" />}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
