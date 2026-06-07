'use client';

import { Database, Loader2, X } from 'lucide-react';

import { useMemo, useState } from 'react';

import type { BIMObject } from '@/5entities/bim-object';
import type { DbVersion } from '@/5entities/db-version';
import type { PredictionSession } from '@/5entities/prediction';
import { createVersion } from '@/6shared/api';
import { useLocale } from '@/6shared/i18n';
import { Badge } from '@/6shared/ui/primitive/badge';
import { Button } from '@/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/6shared/ui/primitive/card';
import { Input } from '@/6shared/ui/primitive/input';
import { Label } from '@/6shared/ui/primitive/label';
import { SelectableCardList } from '@/6shared/ui/SelectableCardList';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/6shared/ui/primitive/table';

import { buildUpdateList } from '../model/buildUpdateList';
import { selectByConfidence } from '../model/selectByConfidence';

interface CreateVersionPanelProps {
  objects: BIMObject[];
  predictionMap: Record<string, PredictionSession[]>;
  selectedVersion: string | undefined;
  versions: DbVersion[];
  onCreated: () => void;
}

// Reserve a stable space of N rows in the update-target table (mirrors the
// object-list panel in 1height/plane-2): short lists pad with filler rows so
// the panel height stays put; longer lists grow past it.
const RESERVED_ROWS = 10;

export function CreateVersionPanel({
  objects,
  predictionMap,
  selectedVersion,
  versions,
  onCreated,
}: CreateVersionPanelProps) {
  const { t } = useLocale();
  const [name, setName] = useState('');
  const [base, setBase] = useState(''); // '' = none
  const [threshold, setThreshold] = useState(70);
  // index → threshold (%) it was added at, so the source column can show it.
  const [manualAdded, setManualAdded] = useState<Map<number, number>>(new Map());
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  const rows = useMemo(
    () =>
      buildUpdateList(objects, predictionMap, selectedVersion, {
        manualAdded,
        dismissed,
      }),
    [objects, predictionMap, selectedVersion, manualAdded, dismissed],
  );

  const handleAddByConfidence = () => {
    const picked = selectByConfidence(
      objects,
      predictionMap,
      selectedVersion,
      threshold,
    );
    setManualAdded((prev) => {
      const next = new Map(prev);
      picked.forEach((i) => next.set(i, threshold));
      return next;
    });
    setDismissed((prev) => {
      const next = new Set(prev);
      picked.forEach((i) => next.delete(i));
      return next;
    });
  };

  const handleRemove = (globalIndex: number) => {
    setDismissed((prev) => new Set(prev).add(globalIndex));
    setManualAdded((prev) => {
      const next = new Map(prev);
      next.delete(globalIndex);
      return next;
    });
  };

  const reset = () => {
    setName('');
    setManualAdded(new Map());
    setDismissed(new Set());
  };

  const handleCreate = async () => {
    setSubmitting(true);
    setMessage(null);
    const res = await createVersion({
      name: name.trim(),
      base: base === '' ? null : base,
      items: rows.map((r) => r.item),
    });
    setSubmitting(false);
    if (res.success && res.data) {
      setIsError(false);
      setMessage(
        t.createVersion.success(res.data.version, res.data.added, res.data.total),
      );
      reset();
      onCreated();
    } else {
      setIsError(true);
      setMessage(res.error ?? 'DB 버전 생성 실패');
    }
  };

  const canCreate =
    name.trim() !== '' &&
    !submitting &&
    (rows.length > 0 || base !== '');

  // Base-DB options for the reused card list: a synthetic "none" entry (empty
  // DB) followed by the real versions. Empty name === the "no base" choice.
  const baseItems: DbVersion[] = [{ name: '', points: 0 }, ...versions];

  return (
    // 2height mirrors 1height: a left control plane (stacked cards) + a right
    // list plane (the update-target table, like the object-list panel).
    <div className="grid grid-cols-[280px_1fr] gap-4">
      {/* Plane 1: DB name + base DB + create action */}
      <div className="flex flex-col gap-4">
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>{t.createVersion.dbName}</CardTitle>
          </CardHeader>
          <CardContent>
            <Input
              id="cv-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t.createVersion.dbNamePlaceholder}
            />
          </CardContent>
        </Card>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>{t.createVersion.baseDb}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-48 overflow-y-auto">
              <SelectableCardList
                items={baseItems}
                getKey={(v) => v.name}
                selectedKey={base}
                onSelect={setBase}
                renderIcon={() => (
                  <Database className="h-7 w-7 shrink-0 text-blue-600" />
                )}
                renderTitle={(v) =>
                  v.name === '' ? t.createVersion.baseNone : v.name
                }
                renderSubtitle={(v) =>
                  v.name === '' ? null : t.version.items(v.points)
                }
              />
            </div>
          </CardContent>
        </Card>

        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>{t.createVersion.title}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button
              className="w-full"
              onClick={handleCreate}
              disabled={!canCreate}
            >
              {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {submitting ? t.createVersion.creating : t.createVersion.create}
            </Button>
            {message && (
              <div
                className={
                  isError
                    ? 'text-destructive text-sm'
                    : 'text-sm text-green-600 dark:text-green-400'
                }
              >
                {message}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Plane 2: update-target list */}
      <Card className="flex flex-col">
        <CardHeader>
          <div className="flex flex-wrap items-end justify-between gap-3">
            <CardTitle>{t.createVersion.targetList(rows.length)}</CardTitle>
            {/* threshold + confidence-add populate the list (not DB metadata) */}
            <div className="flex items-end gap-2">
              <div className="flex flex-col gap-1">
                <Label htmlFor="cv-th" className="text-xs">
                  {t.createVersion.threshold} (%)
                </Label>
                <Input
                  id="cv-th"
                  type="number"
                  min={0}
                  max={100}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value) || 0)}
                  className="h-8 w-20"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleAddByConfidence}
              >
                {t.createVersion.addByConfidence(threshold)}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10 text-center">#</TableHead>
                  <TableHead>{t.createVersion.colName}</TableHead>
                  <TableHead className="w-28">
                    {t.createVersion.colKbims}
                  </TableHead>
                  <TableHead className="w-28">
                    {t.createVersion.colPps}
                  </TableHead>
                  <TableHead className="w-40">
                    {t.createVersion.colSource}
                  </TableHead>
                  <TableHead className="w-12 text-center">
                    {t.createVersion.colRemove}
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="text-muted-foreground py-6 text-center text-sm"
                    >
                      {objects.length === 0
                        ? t.createVersion.noObjects
                        : t.createVersion.empty}
                    </TableCell>
                  </TableRow>
                ) : (
                  rows.map((r, i) => (
                    <TableRow key={r.globalIndex}>
                      <TableCell className="text-muted-foreground text-center">
                        {i + 1}
                      </TableCell>
                      <TableCell className="truncate" title={r.name}>
                        {r.name || '-'}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {r.item.kbims_code || '-'}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {r.item.pps_code || '-'}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          <Badge
                            variant="outline"
                            className="min-w-0 truncate font-mono text-[10px]"
                            title={r.version || undefined}
                          >
                            {r.version || '—'}
                          </Badge>
                          <Badge
                            className="shrink-0"
                            variant={
                              r.source === 'user' ? 'default' : 'secondary'
                            }
                          >
                            {r.source === 'user'
                              ? t.createVersion.sourceUser
                              : t.createVersion.sourceConfidenceThreshold(
                                  r.threshold ?? 0,
                                )}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="text-center">
                        <button
                          type="button"
                          onClick={() => handleRemove(r.globalIndex)}
                          className="text-muted-foreground hover:text-destructive mx-auto block"
                          aria-label={t.createVersion.colRemove}
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
                {/* Pad to RESERVED_ROWS so the table holds a stable height —
                    the empty/placeholder state counts as one occupied row. */}
                {Array.from({
                  length: Math.max(
                    0,
                    RESERVED_ROWS - Math.max(rows.length, 1),
                  ),
                }).map((_, i) => (
                  <TableRow
                    key={`pad-${i}`}
                    aria-hidden
                    className="pointer-events-none"
                  >
                    <TableCell colSpan={6}>
                      <div className="h-5" />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
