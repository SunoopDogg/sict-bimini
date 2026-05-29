'use client';

import type { ReactNode } from 'react';

import { Moon, Settings, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';

import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/6shared/ui/primitive/dropdown-menu';

function SettingItem({
  label,
  onSelect,
  indicator,
}: {
  label: string;
  onSelect: () => void;
  indicator: ReactNode;
}) {
  return (
    <DropdownMenuItem
      onSelect={(e) => {
        e.preventDefault();
        onSelect();
      }}
      className="flex justify-between gap-4"
    >
      <span>{label}</span>
      {indicator}
    </DropdownMenuItem>
  );
}

export function SettingsDropdown() {
  const { locale, setLocale, t } = useLocale();
  const { resolvedTheme, setTheme } = useTheme();
  const ThemeIcon = resolvedTheme === 'dark' ? Moon : Sun;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <SettingItem
          label={t.settings.language}
          onSelect={() => setLocale(locale === 'ko' ? 'en' : 'ko')}
          indicator={
            <span className="text-muted-foreground font-mono text-xs">
              {locale === 'ko' ? 'KO' : 'EN'}
            </span>
          }
        />
        <SettingItem
          label={t.settings.theme}
          onSelect={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
          indicator={<ThemeIcon className="h-3.5 w-3.5" />}
        />
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
