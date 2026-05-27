'use client';

import { useState } from 'react';

import { LocaleContext } from '@/6shared/i18n/context';
import { translations, type Locale } from '@/6shared/i18n/translations';

export function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    if (typeof window === 'undefined') return 'ko';
    return (localStorage.getItem('locale') as Locale) ?? 'ko';
  });

  const setLocale = (l: Locale) => {
    setLocaleState(l);
    localStorage.setItem('locale', l);
  };

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t: translations[locale] }}>
      {children}
    </LocaleContext.Provider>
  );
}
