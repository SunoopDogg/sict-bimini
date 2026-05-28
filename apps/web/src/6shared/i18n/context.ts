'use client';

import { createContext, useContext } from 'react';

import { translations, type Locale, type Translations } from './translations';

interface LocaleContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: Translations;
}

export const LocaleContext = createContext<LocaleContextValue>({
  locale: 'ko',
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setLocale: () => {},
  t: translations.ko,
});

export function useLocale() {
  return useContext(LocaleContext);
}
