import type { Locale } from '@/6shared/i18n';

const DATE_OPTIONS: Intl.DateTimeFormatOptions = {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
};

const dateFormatters: Record<Locale, Intl.DateTimeFormat> = {
  ko: new Intl.DateTimeFormat('ko-KR', DATE_OPTIONS),
  en: new Intl.DateTimeFormat('en-US', DATE_OPTIONS),
};

export function formatDateTime(date: string, locale: Locale = 'ko'): string {
  return dateFormatters[locale].format(new Date(date));
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
