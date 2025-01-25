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

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { PredictionSession } from '@/src/5entities/prediction';
import { cn } from '@/src/6shared/lib/cn';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import { Checkbox } from '@/src/6shared/ui/primitive/checkbox';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from '@/src/6shared/ui/primitive/pagination';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/src/6shared/ui/primitive/table';

interface ObjectListPanelProps {
  selectedFile?: string;
  objects: BIMObjectInput[];
  isLoading?: boolean;
  isPredicting?: boolean;
  predictingIndex?: number | null;
  selectedIndices: Set<number>;
  onSelectionChange: (indices: Set<number>) => void;
  onPredict: () => void;
  onRowClick: (obj: BIMObjectInput, index: number) => void;
  predictionMap: Map<number, PredictionSession[]>;
  focusedIndex: number | null;
}

const PAGE_SIZE = 20;

function KbimsStatusIcon({
  predictionMap,
  index,
  kbimsCode,
}: {
  predictionMap: Map<number, PredictionSession[]>;
  index: number;
  kbimsCode: string;
}) {
  const sessions = predictionMap.get(index);
  const latestSession = sessions?.[sessions.length - 1];
  const prediction = latestSession?.candidates?.[latestSession.selectedIndex];

  if (!prediction) {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900">
        <Minus className="h-3 w-3 text-yellow-600 dark:text-yellow-400" />
      </span>
    );
  }

  if (prediction.predicted_code === kbimsCode) {
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

function PpsStatusIcon({
  predictionMap,
  index,
  ppsCode,
}: {
  predictionMap: Map<number, PredictionSession[]>;
  index: number;
  ppsCode: string;
}) {
  const sessions = predictionMap.get(index);
  const latestSession = sessions?.[sessions.length - 1];
  const prediction = latestSession?.candidates?.[latestSession.selectedIndex];

  if (!prediction) {
    return (
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-yellow-100 dark:bg-yellow-900">
        <Minus className="h-3 w-3 text-yellow-600 dark:text-yellow-400" />
      </span>
    );
  }

  if (prediction.predicted_pps_code === ppsCode) {
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
}: ObjectListPanelProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [prevObjects, setPrevObjects] = useState(objects);

  if (objects !== prevObjects) {
    setPrevObjects(objects);
    setCurrentPage(1);
  }

  const totalPages = Math.ceil(objects.length / PAGE_SIZE);
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const paginatedObjects = objects.slice(startIndex, startIndex + PAGE_SIZE);

  const getPageNumbers = () => {
    const maxVisible = 5;
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
          파일을 선택하면 객체 리스트가 표시됩니다.
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
          객체가 없습니다.
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-4">
        <div className="flex-1 overflow-y-auto">
          <Table>
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
                <TableHead className="text-center">#</TableHead>
                <TableHead>객체 이름</TableHead>
                <TableHead>부위코드</TableHead>
                <TableHead>PPS 코드</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedObjects.map((obj, index) => {
                const globalIndex = startIndex + index;
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
                    <TableCell className="max-w-50 truncate" title={obj.name}>
                      {obj.name || '-'}
                    </TableCell>
                    <TableCell className="max-w-30">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate">
                          {obj.kbims_code || '-'}
                        </span>
                        <KbimsStatusIcon
                          predictionMap={predictionMap}
                          index={globalIndex}
                          kbimsCode={obj.kbims_code}
                        />
                      </div>
                    </TableCell>
                    <TableCell className="max-w-30">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate">{obj.pps_code || '-'}</span>
                        <PpsStatusIcon
                          predictionMap={predictionMap}
                          index={globalIndex}
                          ppsCode={obj.pps_code}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
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
                  aria-label="이전 페이지"
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
                  aria-label="다음 페이지"
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
          <CardTitle>객체 리스트</CardTitle>
          {selectedFile && objects.length > 0 && (
            <Button
              size="sm"
              onClick={onPredict}
              disabled={isPredicting || selectedIndices.size === 0}
            >
              {isPredicting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              선택 예측 ({selectedIndices.size}개)
            </Button>
          )}
        </div>
        {selectedFile && (
          <CardDescription>
            {selectedFile} ({objects.length}개 객체)
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex flex-1 flex-col overflow-hidden">
        {renderContent()}
      </CardContent>
    </Card>
  );
}
