'use client';

import { useState } from 'react';
import { Loader2 } from 'lucide-react';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import { cn } from '@/src/6shared/lib/cn';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
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
  onPredict: () => void;
  onRowClick: (obj: BIMObjectInput, index: number) => void;
}

const PAGE_SIZE = 20;

export function ObjectListPanel({
  selectedFile,
  objects,
  isLoading = false,
  isPredicting = false,
  predictingIndex = null,
  onPredict,
  onRowClick,
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
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    const pages: (number | string)[] = [1];

    if (currentPage > 3) {
      pages.push('...');
    }

    const start = Math.max(2, currentPage - 1);
    const end = Math.min(totalPages - 1, currentPage + 1);

    for (let i = start; i <= end; i++) {
      if (!pages.includes(i)) {
        pages.push(i);
      }
    }

    if (currentPage < totalPages - 2) {
      pages.push('...');
    }

    if (totalPages > 1) {
      pages.push(totalPages);
    }

    return pages;
  };

  const renderContent = () => {
    if (!selectedFile) {
      return (
        <div className="flex items-center justify-center py-12 text-muted-foreground">
          파일을 선택하면 객체 리스트가 표시됩니다.
        </div>
      );
    }

    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      );
    }

    if (objects.length === 0) {
      return (
        <div className="flex items-center justify-center py-12 text-muted-foreground">
          객체가 없습니다.
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-4">
        <div className="flex-1 overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 bg-background">
              <TableRow>
                <TableHead>#</TableHead>
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
                    )}
                  >
                    <TableCell className="text-muted-foreground">
                      {globalIndex + 1}
                      {predictingIndex === globalIndex && (
                        <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />
                      )}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate" title={obj.name}>
                      {obj.name || '-'}
                    </TableCell>
                    <TableCell className="max-w-[120px] truncate">
                      {obj.kbims_code || '-'}
                    </TableCell>
                    <TableCell className="max-w-[120px] truncate">
                      {obj.pps_code || '-'}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>

        {totalPages > 1 && (
          <div className="flex flex-col gap-2">
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={(e) => {
                      e.preventDefault();
                      if (currentPage > 1) setCurrentPage((p) => p - 1);
                    }}
                    className={currentPage === 1 ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                  />
                </PaginationItem>

                {getPageNumbers().map((page, idx) => (
                  <PaginationItem key={typeof page === 'number' ? page : `ellipsis-${idx}`}>
                    {typeof page === 'number' ? (
                      <PaginationLink
                        isActive={currentPage === page}
                        onClick={(e) => {
                          e.preventDefault();
                          setCurrentPage(page);
                        }}
                        className="cursor-pointer"
                      >
                        {page}
                      </PaginationLink>
                    ) : (
                      <PaginationEllipsis />
                    )}
                  </PaginationItem>
                ))}

                <PaginationItem>
                  <PaginationNext
                    onClick={(e) => {
                      e.preventDefault();
                      if (currentPage < totalPages) setCurrentPage((p) => p + 1);
                    }}
                    className={currentPage === totalPages ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>

            <div className="text-center text-xs text-muted-foreground">
              {currentPage} / {totalPages} 페이지
            </div>
          </div>
        )}

        <Button
          onClick={onPredict}
          disabled={isPredicting || objects.length === 0}
          className="w-full"
        >
          {isPredicting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          부위코드 일괄 예측
        </Button>
      </div>
    );
  };

  return (
    <Card className="flex h-full flex-col">
      <CardHeader>
        <CardTitle>객체 리스트</CardTitle>
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
