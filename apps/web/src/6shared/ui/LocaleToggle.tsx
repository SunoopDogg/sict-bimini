'use client';

import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';

export function LocaleToggle() {
  const { locale, setLocale, t } = useLocale();
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => setLocale(locale === 'ko' ? 'en' : 'ko')}
      aria-label={t.localeToggle}
    >
      {locale === 'ko' ? 'EN' : 'KO'}
    </Button>
  );
}
