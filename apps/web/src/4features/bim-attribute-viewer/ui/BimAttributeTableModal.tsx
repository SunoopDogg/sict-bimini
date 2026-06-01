'use client';

import { TableProperties } from 'lucide-react';

import { useEffect, useState } from 'react';

import type { BIMAttributeListResponse } from '@/5entities/bim-attribute';
import { fetchBimAttributes } from '@/6shared/api';
import { useLocale } from '@/6shared/i18n';
import { Button } from '@/6shared/ui/primitive/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/6shared/ui/primitive/dialog';
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/6shared/ui/primitive/pagination';
import { Skeleton } from '@/6shared/ui/primitive/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/6shared/ui/primitive/table';

const PAGE_SIZE = 20;

interface BimAttributeTableModalProps {
  version?: string;
}

export function BimAttributeTableModal({
  version,
}: BimAttributeTableModalProps = {}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [data, setData] = useState<BIMAttributeListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { t } = useLocale();

  const fetchData = async (pageNum: number) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchBimAttributes(pageNum, PAGE_SIZE, version);
      if (response.success && response.data) {
        setData(response.data);
      } else {
        setError(response.error || t.bimAttr.loadFailed);
      }
    } catch {
      setError(t.bimAttr.loadError);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      fetchData(page);
    }
  }, [open, page, version]);

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

    items.push(
      <PaginationItem key={1}>
        <PaginationLink onClick={() => setPage(1)} isActive={current === 1}>
          1
        </PaginationLink>
      </PaginationItem>,
    );

    if (current > 3) {
      items.push(<PaginationEllipsis key="ellipsis-start" />);
    }

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

    if (current < totalPages - 2) {
      items.push(<PaginationEllipsis key="ellipsis-end" />);
    }

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
          {t.bimAttr.trigger}
        </Button>
      </DialogTrigger>
      <DialogContent className="flex h-[80vh] max-w-6xl flex-col">
        <DialogHeader>
          <DialogTitle>{t.bimAttr.title}</DialogTitle>
          <DialogDescription>
            {t.bimAttr.subtitle}
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
                    <TableHead className="w-[12%]">{t.bimAttr.colIfcType}</TableHead>
                    <TableHead className="w-[10%]">{t.bimAttr.colCategory}</TableHead>
                    <TableHead className="w-[12%]">{t.bimAttr.colFamilyName}</TableHead>
                    <TableHead className="w-[12%]">{t.bimAttr.colKbimsCode}</TableHead>
                    <TableHead className="w-[14%]">{t.bimAttr.colPpsCode}</TableHead>
                    <TableHead className="w-[18%]">{t.bimAttr.colFamily}</TableHead>
                    <TableHead className="w-[12%]">{t.bimAttr.colType}</TableHead>
                    <TableHead className="w-[10%]">{t.bimAttr.colTypeId}</TableHead>
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
                        {t.bimAttr.noData}
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
                {data ? t.bimAttr.total(data.total) : ' '}
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
