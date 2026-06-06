'use client';

import { Loader2, X } from 'lucide-react';

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
  const [manualAdded, setManualAdded] = useState<Set<number>>(new Set());
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
    setManualAdded((prev) => new Set([...prev, ...picked]));
    setDismissed((prev) => {
      const next = new Set(prev);
      picked.forEach((i) => next.delete(i));
      return next;
    });
  };

  const handleRemove = (globalIndex: number) => {
    setDismissed((prev) => new Set(prev).add(globalIndex));
    setManualAdded((prev) => {
      const next = new Set(prev);
      next.delete(globalIndex);
      return next;
    });
  };

  const reset = () => {
    setName('');
    setManualAdded(new Set());
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

  return (
    <Card className="mt-4 flex flex-col">
      <CardHeader>
        <CardTitle>{t.createVersion.title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end gap-4">
          <div className="flex flex-col gap-1">
            <Label htmlFor="cv-name">{t.createVersion.dbName}</Label>
            <Input
              id="cv-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t.createVersion.dbNamePlaceholder}
              className="w-64"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="cv-base">{t.createVersion.baseDb}</Label>
            <select
              id="cv-base"
              value={base}
              onChange={(e) => setBase(e.target.value)}
              className="border-input bg-background h-9 rounded-md border px-3 text-sm"
            >
              <option value="">{t.createVersion.baseNone}</option>
              {versions.map((v) => (
                <option key={v.name} value={v.name}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <Label htmlFor="cv-th">{t.createVersion.threshold} (%)</Label>
            <Input
              id="cv-th"
              type="number"
              min={0}
              max={100}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value) || 0)}
              className="w-24"
            />
          </div>
          <Button variant="outline" onClick={handleAddByConfidence}>
            {t.createVersion.addByConfidence(threshold)}
          </Button>
          <Button onClick={handleCreate} disabled={!canCreate}>
            {submitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {submitting ? t.createVersion.creating : t.createVersion.create}
          </Button>
        </div>

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

        {objects.length === 0 ? (
          <div className="text-muted-foreground py-6 text-center text-sm">
            {t.createVersion.noObjects}
          </div>
        ) : (
          <div>
            <div className="text-muted-foreground mb-2 text-sm">
              {t.createVersion.targetList(rows.length)}
            </div>
            {rows.length === 0 ? (
              <div className="text-muted-foreground py-4 text-center text-sm">
                {t.createVersion.empty}
              </div>
            ) : (
              <div className="max-h-64 overflow-y-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-10">#</TableHead>
                      <TableHead>{t.createVersion.colName}</TableHead>
                      <TableHead className="w-28">
                        {t.createVersion.colKbims}
                      </TableHead>
                      <TableHead className="w-28">
                        {t.createVersion.colPps}
                      </TableHead>
                      <TableHead className="w-24">
                        {t.createVersion.colSource}
                      </TableHead>
                      <TableHead className="w-12">
                        {t.createVersion.colRemove}
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rows.map((r, i) => (
                      <TableRow key={r.globalIndex}>
                        <TableCell className="text-muted-foreground">
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
                          <Badge
                            variant={
                              r.source === 'user' ? 'default' : 'secondary'
                            }
                          >
                            {r.source === 'user'
                              ? t.createVersion.sourceUser
                              : t.createVersion.sourceConfidence}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <button
                            type="button"
                            onClick={() => handleRemove(r.globalIndex)}
                            className="text-muted-foreground hover:text-destructive"
                            aria-label={t.createVersion.colRemove}
                          >
                            <X className="h-4 w-4" />
                          </button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
