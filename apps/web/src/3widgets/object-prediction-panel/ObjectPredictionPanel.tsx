'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';

import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type { PredictionResult, PredictionSession } from '@/src/5entities/prediction';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import { Button } from '@/src/6shared/ui/primitive/button';
import { Input } from '@/src/6shared/ui/primitive/input';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from '@/src/6shared/ui/primitive/pagination';
import { cn } from '@/src/6shared/lib/cn';

interface ObjectPredictionPanelProps {
  object: BIMObjectInput | null;
  sessions: PredictionSession[];
  isPredicting: boolean;
  onPredict: () => void;
  onSelectCandidate: (sessionIndex: number, candidateIndex: number) => void;
  onUserCandidateChange: (sessionIndex: number, candidate: PredictionResult) => void;
}

function buildUserCandidate(
  existing: PredictionResult | undefined,
  override: Partial<PredictionResult>,
): PredictionResult {
  return {
    predicted_code: existing?.predicted_code ?? null,
    predicted_pps_code: existing?.predicted_pps_code ?? null,
    reasoning: existing?.reasoning ?? '',
    confidence: 0,
    predicted_at: existing?.predicted_at ?? new Date().toISOString(),
    ...override,
  };
}

function confidenceBadgeClass(confidence: number): string {
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
  onUserCandidateChange,
}: ObjectPredictionPanelProps) {
  const [sessionPage, setSessionPage] = useState(0);
  const [previousSessionCount, setPreviousSessionCount] = useState(sessions.length);

  if (sessions.length !== previousSessionCount) {
    setPreviousSessionCount(sessions.length);
    setSessionPage(0);
  }

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
  const reversedSessions = [...sessions].reverse();
  const currentSession = reversedSessions[sessionPage];
  const currentSessionOriginalIndex = sessions.length - 1 - sessionPage;

  return (
    <Card>
      <CardHeader>
        <CardTitle>예측 결과</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <ObjectInfo object={object} />

          {sessions.length > 1 && (
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationLink
                    onClick={(e) => {
                      e.preventDefault();
                      if (sessionPage > 0) setSessionPage((p) => p - 1);
                    }}
                    aria-label="이전 예측"
                    className={cn(
                      'h-8 w-8 cursor-pointer p-0',
                      sessionPage === 0 && 'pointer-events-none opacity-50',
                    )}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </PaginationLink>
                </PaginationItem>
                <PaginationItem>
                  <span className="flex h-8 items-center px-2 text-sm text-muted-foreground">
                    예측 #{currentSessionOriginalIndex + 1} / {sessions.length}
                  </span>
                </PaginationItem>
                <PaginationItem>
                  <PaginationLink
                    onClick={(e) => {
                      e.preventDefault();
                      if (sessionPage < sessions.length - 1) setSessionPage((p) => p + 1);
                    }}
                    aria-label="다음 예측"
                    className={cn(
                      'h-8 w-8 cursor-pointer p-0',
                      sessionPage === sessions.length - 1 && 'pointer-events-none opacity-50',
                    )}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </PaginationLink>
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}

          {currentSession && (
            <div className="space-y-3 rounded-lg border p-4">
              <div className="flex items-center justify-between">
                <h3 className="text-muted-foreground text-sm font-medium">
                  예측 #{currentSessionOriginalIndex + 1}
                </h3>
                <span className="text-muted-foreground text-xs">
                  {new Date(currentSession.predicted_at).toLocaleString('ko-KR')}
                </span>
              </div>
              <div className="space-y-2">
                {currentSession.candidates.map((candidate, candidateIdx) => (
                  <button
                    key={candidateIdx}
                    type="button"
                    onClick={() => onSelectCandidate(currentSessionOriginalIndex, candidateIdx)}
                    className={cn(
                      'w-full rounded-lg border p-3 text-left transition-colors',
                      currentSession.selectedIndex === candidateIdx
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
                          className={confidenceBadgeClass(candidate.confidence)}
                        >
                          {(candidate.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      {currentSession.selectedIndex === candidateIdx && (
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
                {/* User input card */}
                <button
                  type="button"
                  onClick={() => onSelectCandidate(currentSessionOriginalIndex, currentSession.candidates.length)}
                  className={cn(
                    'w-full rounded-lg border p-3 text-left transition-colors',
                    currentSession.selectedIndex === currentSession.candidates.length
                      ? 'border-primary bg-primary/5'
                      : 'hover:bg-muted/50',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-muted-foreground text-xs font-medium">
                        사용자 입력
                      </span>
                      <Badge className="bg-gray-100 text-gray-800">
                        직접 입력
                      </Badge>
                    </div>
                    {currentSession.selectedIndex === currentSession.candidates.length && (
                      <span className="text-primary text-xs font-medium">
                        선택됨
                      </span>
                    )}
                  </div>
                  <div className="mt-2 space-y-2" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">부위코드</span>
                      <Input
                        value={currentSession.userCandidate?.predicted_code ?? ''}
                        onChange={(e) => {
                          onUserCandidateChange(
                            currentSessionOriginalIndex,
                            buildUserCandidate(currentSession.userCandidate, { predicted_code: e.target.value }),
                          );
                        }}
                        placeholder="부위코드 입력"
                        className="h-7 text-sm"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">PPS 코드</span>
                      <Input
                        value={currentSession.userCandidate?.predicted_pps_code ?? ''}
                        onChange={(e) => {
                          onUserCandidateChange(
                            currentSessionOriginalIndex,
                            buildUserCandidate(currentSession.userCandidate, { predicted_pps_code: e.target.value }),
                          );
                        }}
                        placeholder="PPS 코드 입력"
                        className="h-7 text-sm"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">설명</span>
                      <Input
                        value={currentSession.userCandidate?.reasoning ?? ''}
                        onChange={(e) => {
                          onUserCandidateChange(
                            currentSessionOriginalIndex,
                            buildUserCandidate(currentSession.userCandidate, { reasoning: e.target.value }),
                          );
                        }}
                        placeholder="설명 입력"
                        className="h-7 text-sm"
                      />
                    </div>
                  </div>
                </button>
              </div>
            </div>
          )}

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
