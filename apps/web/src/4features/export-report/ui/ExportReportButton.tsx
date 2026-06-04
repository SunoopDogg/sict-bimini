'use client';

import { useState } from 'react';

import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import { fetchMeta } from '@/6shared/api';
import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';

import { buildReportData } from '../model/buildReportData';

interface Props {
  objects: BIMObject[];
  predictionMap: Record<string, PredictionSession[]>;
  // Predicts any not-yet-predicted objects and resolves to the fresh map.
  // When omitted, the report is built from `predictionMap` as-is.
  onEnsureAllPredicted?: () => Promise<Record<string, PredictionSession[]>>;
  version?: string;
  fileName?: string;
  className?: string;
}

export function ExportReportButton({
  objects,
  predictionMap,
  onEnsureAllPredicted,
  version,
  fileName,
  className,
}: Props) {
  const { t } = useLocale();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const handleClick = async () => {
    setError(undefined);
    setBusy(true);
    try {
      // Predict any missing objects first (skips already-predicted ones), then
      // build the report from the fresh map the predict step returns.
      const map = onEnsureAllPredicted
        ? await onEnsureAllPredicted()
        : predictionMap;
      const meta = await fetchMeta();
      const generatedAt = new Date().toLocaleString('ko-KR', {
        timeZone: 'Asia/Seoul',
      });
      const data = buildReportData(objects, map, {
        version,
        fileName,
        llmModel: meta.success && meta.data ? meta.data.llm_model : 'N/A',
        embeddingModel:
          meta.success && meta.data ? meta.data.embedding_model : 'N/A',
        generatedAt,
      });
      const { downloadReportPdf } = await import('../lib/generatePdf');
      const base =
        (fileName?.replace(/\.[^.]+$/, '') || 'predict') + '_' + Date.now();
      await downloadReportPdf(data, base);
    } catch (e) {
      setError(e instanceof Error ? e.message : t.report.failed);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      className={className}
      onClick={handleClick}
      disabled={busy || objects.length === 0}
      title={objects.length === 0 ? t.report.noObjects : undefined}
    >
      {busy ? t.report.generating : t.report.export}
      {error && <span className="ml-2 text-xs text-red-500">!</span>}
    </Button>
  );
}
