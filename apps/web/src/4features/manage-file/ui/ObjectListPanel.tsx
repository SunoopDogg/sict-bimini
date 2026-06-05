'use client';

import {
  Check,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Minus,
  X,
} from 'lucide-react';

import { useState } from 'react';

import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import { findSessionForVersion, getSelectedPrediction } from '@/5entities/prediction';
import { useLocale } from '@/6shared/i18n';
import { cn } from '@/6shared/lib/cn';
import { Button } from '@/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/6shared/ui/primitive/card';
import { Checkbox } from '@/6shared/ui/primitive/checkbox';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from '@/6shared/ui/primitive/pagination';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/6shared/ui/primitive/table';

interface ObjectListPanelProps {
  selectedFile?: string;
  objects: BIMObject[];
  isLoading?: boolean;
  isPredicting?: boolean;
  predictingIndex?: number | null;
  selectedIndices: Set<number>;
  onSelectionChange: (indices: Set<number>) => void;
  onPredict: () => void;
  onRowClick: (obj: BIMObject, index: number) => void;
  predictionMap: Record<string, PredictionSession[]>;
  focusedIndex: number | null;
  /** Match icon reflects the session predicted against this DB version. */
  selectedVersion?: string;
}

const PAGE_SIZE = 20;
// Column count of the object table — keep in sync with the TableHeader cells
// below; used by the filler rows' colSpan so they span the full width.
const COLUMN_COUNT = 7;

// Stable reference for rows with no sessions yet — avoids allocating a fresh
// empty array on every render of every unpredicted row.
const EMPTY_SESSIONS: PredictionSession[] = [];

function PredictionResultCell({
  session,
  actualCode,
  target,
}: {
  session: PredictionSession | null;
  actualCode: string;
  target: 'kbims' | 'pps';
}) {
  const notPredicted = (
    <span className="text-muted-foreground inline-flex items-center justify-center">
      <Minus className="h-3 w-3" />
    </span>
  );

  if (!session || !session.prediction) {
    return notPredicted;
  }

  const sel = getSelectedPrediction(session);
  const predictedCode = target === 'kbims' ? sel.kbims_code : sel.pps_code;

  if (!predictedCode) {
    return notPredicted;
  }

  // No ground truth → can't judge; show the predicted code instead.
  if (!actualCode) {
    return (
      <span
        className="min-w-0 truncate font-mono text-xs text-blue-600 dark:text-blue-400"
        title={predictedCode}
      >
        {predictedCode}
      </span>
    );
  }

  if (predictedCode === actualCode) {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-green-100 dark:bg-green-900">
        <Check className="h-3 w-3 text-green-600 dark:text-green-400" />
      </span>
    );
  }

  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-red-100 dark:bg-red-900">
      <X className="h-3 w-3 text-red-600 dark:text-red-400" />
    </span>
  );
}

export function ObjectListPanel({
  selectedFile,
  objects,
  isLoading = false,
  isPredicting = false,
  predictingIndex = null,
  selectedIndices,
  onSelectionChange,
  onPredict,
  onRowClick,
  predictionMap,
  focusedIndex,
  selectedVersion,
}: ObjectListPanelProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [previousObjects, setPrevObjects] = useState(objects);
  const { t } = useLocale();

  if (objects !== previousObjects) {
    setPrevObjects(objects);
    setCurrentPage(1);
  }

  const totalPages = Math.ceil(objects.length / PAGE_SIZE);
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const paginatedObjects = objects.slice(startIndex, startIndex + PAGE_SIZE);

  const getPageNumbers = () => {
    const maxVisible = 10;
    if (totalPages <= maxVisible) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }
    const half = Math.floor(maxVisible / 2);
    let start = Math.max(1, currentPage - half);
    const end = Math.min(totalPages, start + maxVisible - 1);
    start = Math.max(1, end - maxVisible + 1);
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  };

  const renderContent = () => {
    if (!selectedFile) {
      return (
        <div className="text-muted-foreground flex items-center justify-center py-12">
          {t.file.selectFilePrompt}
        </div>
      );
    }

    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="text-muted-foreground h-6 w-6 animate-spin" />
        </div>
      );
    }

    if (objects.length === 0) {
      return (
        <div className="text-muted-foreground flex items-center justify-center py-12">
          {t.file.noObjects}
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-4">
        <div className="flex-1 overflow-y-auto">
          <Table className="table-fixed">
            <TableHeader className="bg-muted [&_th]:text-muted-foreground sticky top-0">
              <TableRow>
                <TableHead className="w-10">
                  <Checkbox
                    checked={
                      paginatedObjects.length > 0 &&
                      paginatedObjects.every((_, i) =>
                        selectedIndices.has(startIndex + i),
                      )
                    }
                    onCheckedChange={(checked) => {
                      const next = new Set(selectedIndices);
                      if (checked) {
                        paginatedObjects.forEach((_, i) =>
                          next.add(startIndex + i),
                        );
                      } else {
                        paginatedObjects.forEach((_, i) =>
                          next.delete(startIndex + i),
                        );
                      }
                      onSelectionChange(next);
                    }}
                  />
                </TableHead>
                <TableHead className="w-12 text-center">#</TableHead>
                <TableHead>{t.object.colName}</TableHead>
                <TableHead className="w-24">{t.object.colPartCode}</TableHead>
                <TableHead className="w-24">
                  {t.object.colPredPartCode}
                </TableHead>
                <TableHead className="w-24">{t.object.colPpsCode}</TableHead>
                <TableHead className="w-24">
                  {t.object.colPredPpsCode}
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedObjects.map((obj, index) => {
                const globalIndex = startIndex + index;
                // Resolve the selected-version session once per row; both code
                // columns below render from the same session.
                const versionSession = findSessionForVersion(
                  predictionMap[globalIndex] ?? EMPTY_SESSIONS,
                  selectedVersion,
                );
                return (
                  <TableRow
                    key={globalIndex}
                    onClick={() => onRowClick(obj, globalIndex)}
                    className={cn(
                      'cursor-pointer',
                      predictingIndex === globalIndex && 'bg-muted',
                      focusedIndex === globalIndex && 'bg-accent',
                    )}
                  >
                    <TableCell
                      className="w-10"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Checkbox
                        checked={selectedIndices.has(globalIndex)}
                        onCheckedChange={(checked) => {
                          const next = new Set(selectedIndices);
                          if (checked) {
                            next.add(globalIndex);
                          } else {
                            next.delete(globalIndex);
                          }
                          onSelectionChange(next);
                        }}
                      />
                    </TableCell>
                    <TableCell className="text-muted-foreground text-center">
                      {globalIndex + 1}
                      {predictingIndex === globalIndex && (
                        <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />
                      )}
                    </TableCell>
                    <TableCell className="truncate" title={obj.name}>
                      {obj.name || '-'}
                    </TableCell>
                    <TableCell className="truncate">
                      {obj.kbims_code || '-'}
                    </TableCell>
                    <TableCell>
                      <div className="flex min-w-0 items-center">
                        <PredictionResultCell
                          session={versionSession}
                          actualCode={obj.kbims_code}
                          target="kbims"
                        />
                      </div>
                    </TableCell>
                    <TableCell className="truncate">
                      {obj.pps_code || '-'}
                    </TableCell>
                    <TableCell>
                      <div className="flex min-w-0 items-center">
                        <PredictionResultCell
                          session={versionSession}
                          actualCode={obj.pps_code}
                          target="pps"
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
              {/* Pad short pages to PAGE_SIZE so the table height — and thus the
                  pagination bar below — stays put when flipping pages. */}
              {totalPages > 1 &&
                Array.from({
                  length: PAGE_SIZE - paginatedObjects.length,
                }).map((_, i) => (
                  <TableRow
                    key={`pad-${i}`}
                    aria-hidden
                    className="pointer-events-none"
                  >
                    <TableCell colSpan={COLUMN_COUNT}>
                      <div className="h-5" />
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </div>

        {totalPages > 1 && (
          <Pagination>
            <PaginationContent>
              <PaginationItem>
                <PaginationLink
                  onClick={(e) => {
                    e.preventDefault();
                    if (currentPage > 1) setCurrentPage((p) => p - 1);
                  }}
                  aria-label={t.file.prevPage}
                  className={cn(
                    'h-8 w-8 cursor-pointer p-0',
                    currentPage === 1 && 'pointer-events-none opacity-50',
                  )}
                >
                  <ChevronLeft className="h-4 w-4" />
                </PaginationLink>
              </PaginationItem>

              {getPageNumbers().map((page) => (
                <PaginationItem key={page}>
                  <PaginationLink
                    isActive={currentPage === page}
                    onClick={(e) => {
                      e.preventDefault();
                      setCurrentPage(page);
                    }}
                    className="h-8 w-8 cursor-pointer p-0"
                  >
                    {page}
                  </PaginationLink>
                </PaginationItem>
              ))}

              <PaginationItem>
                <PaginationLink
                  onClick={(e) => {
                    e.preventDefault();
                    if (currentPage < totalPages) setCurrentPage((p) => p + 1);
                  }}
                  aria-label={t.file.nextPage}
                  className={cn(
                    'h-8 w-8 cursor-pointer p-0',
                    currentPage === totalPages &&
                      'pointer-events-none opacity-50',
                  )}
                >
                  <ChevronRight className="h-4 w-4" />
                </PaginationLink>
              </PaginationItem>
            </PaginationContent>
          </Pagination>
        )}
      </div>
    );
  };

  return (
    <Card className="flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{t.file.objectList}</CardTitle>
          {selectedFile && objects.length > 0 && (
            <Button
              size="sm"
              onClick={onPredict}
              disabled={isPredicting || selectedIndices.size === 0}
            >
              {isPredicting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              {t.predict.batchPredict(selectedIndices.size)}
            </Button>
          )}
        </div>
        {selectedFile && (
          <CardDescription>
            {selectedFile} ({t.file.objects(objects.length)})
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex flex-1 flex-col overflow-hidden">
        {renderContent()}
      </CardContent>
    </Card>
  );
}
