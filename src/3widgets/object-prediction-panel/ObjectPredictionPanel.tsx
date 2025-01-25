'use client';

import { Loader2 } from 'lucide-react';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { PredictionSession } from '@/src/5entities/prediction';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import { cn } from '@/src/6shared/lib/cn';

interface ObjectPredictionPanelProps {
  object: BIMObjectInput | null;
  sessions: PredictionSession[];
  isPredicting: boolean;
  onPredict: () => void;
  onSelectCandidate: (sessionIndex: number, candidateIndex: number) => void;
}

function getConfidenceColorClass(confidence: number): string {
  if (confidence >= 0.7)
    return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  if (confidence >= 0.4)
    return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
  return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
}

function ObjectInfo({ object }: { object: BIMObjectInput }) {
  return (
    <div className="space-y-2 rounded-lg border p-4">
      <h3 className="text-muted-foreground text-sm font-medium">객체 정보</h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">이름</dt>
        <dd>{object.name || '-'}</dd>
        <dt className="text-muted-foreground">유형</dt>
        <dd>{object.object_type || '-'}</dd>
        <dt className="text-muted-foreground">카테고리</dt>
        <dd>{object.category || '-'}</dd>
        <dt className="text-muted-foreground">패밀리</dt>
        <dd>{object.family_name || '-'}</dd>
        <dt className="text-muted-foreground">부위코드</dt>
        <dd>{object.kbims_code || '-'}</dd>
        <dt className="text-muted-foreground">PPS 코드</dt>
        <dd>{object.pps_code || '-'}</dd>
      </dl>
    </div>
  );
}

export function ObjectPredictionPanel({
  object,
  sessions,
  isPredicting,
  onPredict,
  onSelectCandidate,
}: ObjectPredictionPanelProps) {
  // State 1: No object selected
  if (object === null) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>예측 결과</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <p className="text-muted-foreground text-center">
              객체를 선택하면 예측 결과가 표시됩니다.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // State 2: Object selected, no predictions yet
  if (sessions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>예측 결과</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <ObjectInfo object={object} />
            <div className="flex justify-center">
              <Button onClick={onPredict} disabled={isPredicting}>
                {isPredicting && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                예측하기
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // State 3: Object selected with prediction results
  return (
    <Card>
      <CardHeader>
        <CardTitle>예측 결과</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <ObjectInfo object={object} />
          {[...sessions].reverse().map((session, reverseIdx) => {
            const sessionIndex = sessions.length - 1 - reverseIdx;
            return (
              <div key={reverseIdx} className="space-y-3 rounded-lg border p-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-muted-foreground text-sm font-medium">
                    예측 #{sessionIndex + 1}
                  </h3>
                  <span className="text-muted-foreground text-xs">
                    {new Date(session.predicted_at).toLocaleString('ko-KR')}
                  </span>
                </div>
                <div className="space-y-2">
                  {session.candidates.map((candidate, candidateIdx) => (
                    <button
                      key={candidateIdx}
                      type="button"
                      onClick={() => onSelectCandidate(sessionIndex, candidateIdx)}
                      className={cn(
                        'w-full rounded-lg border p-3 text-left transition-colors',
                        session.selectedIndex === candidateIdx
                          ? 'border-primary bg-primary/5'
                          : 'hover:bg-muted/50',
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-muted-foreground text-xs font-medium">
                            {candidateIdx + 1}순위
                          </span>
                          <Badge
                            className={getConfidenceColorClass(candidate.confidence)}
                          >
                            {(candidate.confidence * 100).toFixed(0)}%
                          </Badge>
                        </div>
                        {session.selectedIndex === candidateIdx && (
                          <span className="text-primary text-xs font-medium">
                            선택됨
                          </span>
                        )}
                      </div>
                      <div className="mt-2 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground text-xs w-14">부위코드</span>
                          <span className="text-sm font-semibold">
                            {candidate.predicted_code || '-'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground text-xs w-14">PPS 코드</span>
                          <span className="text-sm font-semibold">
                            {candidate.predicted_pps_code || '-'}
                          </span>
                        </div>
                      </div>
                      {candidate.reasoning && (
                        <p className="text-muted-foreground mt-2 text-xs line-clamp-2">
                          {candidate.reasoning}
                        </p>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
          <div className="flex justify-center">
            <Button onClick={onPredict} disabled={isPredicting}>
              {isPredicting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              다시 예측하기
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
