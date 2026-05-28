'use client';

import { Moon, Settings, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/6shared/ui/primitive/dropdown-menu';

export function SettingsDropdown() {
  const { locale, setLocale, t } = useLocale();
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            setLocale(locale === 'ko' ? 'en' : 'ko');
          }}
          className="flex justify-between gap-4"
        >
          <span>{t.settings.language}</span>
          <span className="text-muted-foreground font-mono text-xs">
            {locale === 'ko' ? 'KO' : 'EN'}
          </span>
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            if (mounted) setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
          }}
          className="flex justify-between gap-4"
        >
          <span>{t.settings.theme}</span>
          {mounted &&
            (resolvedTheme === 'dark' ? (
              <Moon className="h-3.5 w-3.5" />
            ) : (
              <Sun className="h-3.5 w-3.5" />
            ))}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
