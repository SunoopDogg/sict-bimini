'use client';

import type { BatchPredictResult } from '@/src/5entities/prediction';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/src/6shared/ui/primitive/table';

interface BatchPredictionResultWidgetProps {
  result?: BatchPredictResult | null;
}

function getConfidenceColorClass(confidence: number): string {
  if (confidence >= 0.7) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  if (confidence >= 0.4) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
  return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
}

function getConfidenceBadge(confidence: number) {
  return (
    <Badge className={getConfidenceColorClass(confidence)}>
      {(confidence * 100).toFixed(0)}%
    </Badge>
  );
}

export function BatchPredictionResultWidget({
  result,
}: BatchPredictionResultWidgetProps) {
  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>예측 결과</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <p className="text-muted-foreground text-center">
              예측 실행 전입니다. 객체 리스트에서 예측을 실행해주세요.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>예측 결과</CardTitle>
        <CardDescription>
          총 {result.total}개 중 성공 {result.successful}개, 실패{' '}
          {result.failed}개
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-[600px] overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 bg-background">
              <TableRow>
                <TableHead>#</TableHead>
                <TableHead>예측 코드</TableHead>
                <TableHead>신뢰도</TableHead>
                <TableHead>추론</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {result.results.map((item, index) => {
                return (
                  <TableRow key={index}>
                    <TableCell className="text-muted-foreground">
                      {index + 1}
                    </TableCell>
                    <TableCell>
                      {item.error ? (
                        <Badge variant="destructive">오류</Badge>
                      ) : item.prediction?.predicted_code ? (
                        <span className="font-medium">
                          {item.prediction.predicted_code}
                        </span>
                      ) : (
                        '-'
                      )}
                    </TableCell>
                    <TableCell>
                      {item.prediction
                        ? getConfidenceBadge(item.prediction.confidence)
                        : '-'}
                    </TableCell>
                    <TableCell className="max-w-[300px]">
                      {item.prediction?.reasoning || '-'}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
