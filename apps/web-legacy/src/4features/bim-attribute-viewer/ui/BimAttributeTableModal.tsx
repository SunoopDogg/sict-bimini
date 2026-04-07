'use client';

import { TableProperties } from 'lucide-react';

import { useEffect, useState } from 'react';

import type { BIMAttributeListResponse } from '@/src/5entities/bim-attribute';
import { fetchBimAttributes } from '@/src/6shared/api';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/src/6shared/ui/primitive/dialog';
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/src/6shared/ui/primitive/pagination';
import { Skeleton } from '@/src/6shared/ui/primitive/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/src/6shared/ui/primitive/table';

const PAGE_SIZE = 20;

export function BimAttributeTableModal() {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [data, setData] = useState<BIMAttributeListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async (pageNum: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchBimAttributes(pageNum, PAGE_SIZE);
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || '데이터를 불러오는데 실패했습니다.');
      }
    } catch {
      setError('데이터를 불러오는 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchData(page);
    }
  }, [open, page]);

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    if (newOpen) {
      setPage(1);
    }
  };

  const renderPageNumbers = () => {
    if (!data) return null;

    const totalPages = data.total_pages;
    const current = page;
    const items = [];

    // Always show first page
    items.push(
      <PaginationItem key={1}>
        <PaginationLink onClick={() => setPage(1)} isActive={current === 1}>
          1
        </PaginationLink>
      </PaginationItem>,
    );

    // Show ellipsis if needed
    if (current > 3) {
      items.push(<PaginationEllipsis key="ellipsis-start" />);
    }

    // Show pages around current
    const start = Math.max(2, current - 1);
    const end = Math.min(totalPages - 1, current + 1);

    for (let i = start; i <= end; i++) {
      items.push(
        <PaginationItem key={i}>
          <PaginationLink onClick={() => setPage(i)} isActive={current === i}>
            {i}
          </PaginationLink>
        </PaginationItem>,
      );
    }

    // Show ellipsis if needed
    if (current < totalPages - 2) {
      items.push(<PaginationEllipsis key="ellipsis-end" />);
    }

    // Always show last page if there's more than 1 page
    if (totalPages > 1) {
      items.push(
        <PaginationItem key={totalPages}>
          <PaginationLink
            onClick={() => setPage(totalPages)}
            isActive={current === totalPages}
          >
            {totalPages}
          </PaginationLink>
        </PaginationItem>,
      );
    }

    return items;
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <TableProperties className="mr-2 h-4 w-4" />
          속성 데이터
        </Button>
      </DialogTrigger>
      <DialogContent className="flex h-[80vh] max-w-6xl flex-col">
        <DialogHeader>
          <DialogTitle>BIM 속성 데이터 (CSV)</DialogTitle>
          <DialogDescription>
            벡터 데이터베이스에 저장된 BIM 속성 목록입니다.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="border-destructive bg-destructive/10 rounded-lg border p-4">
            <p className="text-destructive text-sm">{error}</p>
          </div>
        )}

        {!error && (loading || data) && (
          <>
            <div className="flex-1 overflow-auto rounded-md border">
              <Table className="table-fixed">
                <TableHeader>
                  <TableRow className="bg-muted">
                    <TableHead className="w-[12%]">IFC 타입</TableHead>
                    <TableHead className="w-[10%]">분류</TableHead>
                    <TableHead className="w-[12%]">패밀리명</TableHead>
                    <TableHead className="w-[12%]">KBIMS 코드</TableHead>
                    <TableHead className="w-[14%]">PPS 코드</TableHead>
                    <TableHead className="w-[18%]">패밀리</TableHead>
                    <TableHead className="w-[12%]">유형</TableHead>
                    <TableHead className="w-[10%]">유형ID</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {loading ? (
                    Array.from({ length: PAGE_SIZE }).map((_, i) => (
                      <TableRow key={`skeleton-${i}`} className="border-border">
                        {Array.from({ length: 8 }).map((_, j) => (
                          <TableCell key={j}>
                            <Skeleton className="h-4 w-full bg-muted" />
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : data && data.items.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={8}
                        className="text-center text-muted-foreground"
                      >
                        데이터가 없습니다.
                      </TableCell>
                    </TableRow>
                  ) : (
                    data?.items.map((item, index) => (
                      <TableRow key={`${item.type_id}-${index}`} className="border-border">
                        <TableCell className="truncate">{item.ifc_type}</TableCell>
                        <TableCell className="truncate">{item.category}</TableCell>
                        <TableCell className="truncate">{item.family_name}</TableCell>
                        <TableCell className="truncate">{item.kbims_code}</TableCell>
                        <TableCell className="truncate">{item.pps_code}</TableCell>
                        <TableCell className="truncate">{item.family}</TableCell>
                        <TableCell className="truncate">{item.type}</TableCell>
                        <TableCell className="truncate">{item.type_id}</TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>

            <div className="flex items-center justify-between pt-4">
              <p className="w-16 text-sm text-muted-foreground">
                {data ? `총 ${data.total.toLocaleString()}건` : '\u00A0'}
              </p>
              {data && (
                <Pagination>
                  <PaginationContent>
                    <PaginationItem>
                      <PaginationPrevious
                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                        className={
                          page === 1
                            ? 'pointer-events-none opacity-50'
                            : 'cursor-pointer'
                        }
                      />
                    </PaginationItem>
                    {renderPageNumbers()}
                    <PaginationItem>
                      <PaginationNext
                        onClick={() =>
                          setPage((p) => Math.min(data.total_pages, p + 1))
                        }
                        className={
                          page === data.total_pages
                            ? 'pointer-events-none opacity-50'
                            : 'cursor-pointer'
                        }
                      />
                    </PaginationItem>
                  </PaginationContent>
                </Pagination>
              )}
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
