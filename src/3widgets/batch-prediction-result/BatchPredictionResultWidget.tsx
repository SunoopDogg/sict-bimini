'use client';

import type { BatchPredictResult } from '@/src/5entities/prediction';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';

interface BatchPredictionResultWidgetProps {
  result: BatchPredictResult;
  onReset?: () => void;
}

function getConfidenceBadge(confidence: number) {
  if (confidence >= 0.7) {
    return (
      <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
        {(confidence * 100).toFixed(0)}%
      </Badge>
    );
  }
  if (confidence >= 0.4) {
    return (
      <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
        {(confidence * 100).toFixed(0)}%
      </Badge>
    );
  }
  return (
    <Badge className="bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
      {(confidence * 100).toFixed(0)}%
    </Badge>
  );
}

export function BatchPredictionResultWidget({
  result,
  onReset,
}: BatchPredictionResultWidgetProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>예측 결과</CardTitle>
            <CardDescription>
              총 {result.total}개 중 성공 {result.successful}개, 실패{' '}
              {result.failed}개
            </CardDescription>
          </div>
          <Button variant="outline" onClick={onReset}>
            새로운 예측
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="px-2 py-3 text-left font-medium">#</th>
                <th className="px-2 py-3 text-left font-medium">Object Type</th>
                <th className="px-2 py-3 text-left font-medium">카테고리</th>
                <th className="px-2 py-3 text-left font-medium">패밀리</th>
                <th className="px-2 py-3 text-left font-medium">유형</th>
                <th className="px-2 py-3 text-left font-medium">PPS 코드</th>
                <th className="px-2 py-3 text-left font-medium">예측 코드</th>
                <th className="px-2 py-3 text-left font-medium">신뢰도</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((item, index) => {
                return (
                  <tr key={index} className="border-b last:border-b-0">
                    <td className="text-muted-foreground px-2 py-3">
                      {index + 1}
                    </td>
                    <td className="max-w-[100px] truncate px-2 py-3">
                      {item.input.object_type}
                    </td>
                    <td className="max-w-[100px] truncate px-2 py-3">
                      {item.input.category}
                    </td>
                    <td className="max-w-[150px] truncate px-2 py-3">
                      {item.input.family}
                    </td>
                    <td className="max-w-[200px] truncate px-2 py-3">
                      {item.input.type}
                    </td>
                    <td className="max-w-[100px] truncate px-2 py-3">
                      {item.input.pps_code || '-'}
                    </td>
                    <td className="px-2 py-3">
                      {item.error ? (
                        <Badge variant="destructive">오류</Badge>
                      ) : item.prediction?.predicted_code ? (
                        <span className="font-medium">
                          {item.prediction.predicted_code}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="px-2 py-3">
                      {item.prediction
                        ? getConfidenceBadge(item.prediction.confidence)
                        : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
