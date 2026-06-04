'use client';

import { Loader2 } from 'lucide-react';
import { useRef, useState } from 'react';

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
  // `onProgress` reports live "done / total" while predicting. When omitted,
  // the report is built from `predictionMap` as-is.
  // `shouldCancel` is polled before each predict chunk; returning true stops
  // the loop at the next chunk boundary (in-flight chunk still completes).
  onEnsureAllPredicted?: (
    onProgress?: (done: number, total: number) => void,
    shouldCancel?: () => boolean,
  ) => Promise<Record<string, PredictionSession[]>>;
  // Notifies the parent while the whole export (predict + PDF) runs, so it can
  // lock file/version switching to avoid a mid-flight data-source race.
  onBusyChange?: (busy: boolean) => void;
  version?: string;
  fileName?: string;
  className?: string;
}

export function ExportReportButton({
  objects,
  predictionMap,
  onEnsureAllPredicted,
  onBusyChange,
  version,
  fileName,
  className,
}: Props) {
  const { t } = useLocale();
  const [busy, setBusy] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number }>();
  const [error, setError] = useState<string>();
  const cancelRef = useRef(false);

  const handleClick = async () => {
    // While running, the button acts as a cancel toggle (cooperative — the
    // predict loop checks cancelRef before each chunk, so it stops at the next
    // boundary, not instantly; show "중단 중…" so the click is acknowledged).
    if (busy) {
      if (cancelling) return;
      cancelRef.current = true;
      setCancelling(true);
      return;
    }
    setError(undefined);
    setProgress(undefined);
    cancelRef.current = false;
    setCancelling(false);
    setBusy(true);
    onBusyChange?.(true);
    try {
      // Predict any missing objects first (skips already-predicted ones), then
      // build the report from the fresh map the predict step returns.
      const map = onEnsureAllPredicted
        ? await onEnsureAllPredicted(
            (done, total) => setProgress({ done, total }),
            () => cancelRef.current,
          )
        : predictionMap;
      if (cancelRef.current) return; // cancelled mid-predict — skip the PDF
      setProgress(undefined); // predict done — PDF phase shows "generating"
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
      setProgress(undefined);
      setCancelling(false);
      onBusyChange?.(false);
    }
  };

  // Predict phase shows a cancellable "중단 (done/total)"; after a cancel click
  // "중단 중…" until the in-flight chunk ends; the brief PDF phase "생성 중…".
  const label = !busy
    ? t.report.export
    : cancelling
      ? t.report.cancelling
      : progress
        ? `${t.report.cancel} (${progress.done}/${progress.total})`
        : t.report.generating;

  return (
    <Button
      variant="outline"
      size="sm"
      className={className}
      onClick={handleClick}
      disabled={objects.length === 0 || (busy && (cancelling || !progress))}
      title={objects.length === 0 ? t.report.noObjects : undefined}
    >
      {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {label}
      {error && <span className="ml-2 text-xs text-red-500">!</span>}
    </Button>
  );
}
