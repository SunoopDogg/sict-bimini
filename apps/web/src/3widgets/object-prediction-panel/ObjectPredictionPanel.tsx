'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';

import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession, UserCandidate } from '@/5entities/prediction';
import { getPairCount } from '@/5entities/prediction';
import { useLocale } from '@/6shared/i18n';
import { Badge } from '@/6shared/ui/primitive/badge';
import { Button } from '@/6shared/ui/primitive/button';
import { Input } from '@/6shared/ui/primitive/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/6shared/ui/primitive/card';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from '@/6shared/ui/primitive/pagination';
import { cn } from '@/6shared/lib/cn';

interface ObjectPredictionPanelProps {
  object: BIMObject | null;
  sessions: PredictionSession[];
  isPredicting: boolean;
  onPredict: () => void;
  onSelectCandidate: (sessionIndex: number, candidateIndex: number) => void;
  onUserCandidateChange: (
    sessionIndex: number,
    candidate: UserCandidate,
  ) => void;
}

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.7)
    return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
  if (confidence >= 0.4)
    return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
  return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
}

function ObjectInfo({ object }: { object: BIMObject }) {
  const { t } = useLocale();
  return (
    <div className="space-y-2 rounded-lg border p-4">
      <h3 className="text-muted-foreground text-sm font-medium">{t.object.info}</h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">{t.object.name}</dt>
        <dd>{object.name || '-'}</dd>
        <dt className="text-muted-foreground">{t.object.type}</dt>
        <dd>{object.ifc_type || '-'}</dd>
        <dt className="text-muted-foreground">{t.object.category}</dt>
        <dd>{object.category || '-'}</dd>
        <dt className="text-muted-foreground">{t.object.family}</dt>
        <dd>{object.family_name || '-'}</dd>
        <dt className="text-muted-foreground">{t.object.partCode}</dt>
        <dd>{object.kbims_code || '-'}</dd>
        <dt className="text-muted-foreground">{t.object.ppsCode}</dt>
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
  const { t } = useLocale();

  if (sessions.length !== previousSessionCount) {
    setPreviousSessionCount(sessions.length);
    setSessionPage(0);
  }

  if (object === null) {
    return (
      <Card>
        <CardHeader><CardTitle>{t.predict.results}</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12">
            <p className="text-muted-foreground text-center">{t.predict.selectObjectPrompt}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (sessions.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>{t.predict.results}</CardTitle></CardHeader>
        <CardContent>
          <div className="space-y-4">
            <ObjectInfo object={object} />
            <div className="flex justify-center">
              <Button onClick={onPredict} disabled={isPredicting}>
                {isPredicting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {t.predict.predict}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const reversedSessions = [...sessions].reverse();
  const currentSession = reversedSessions[sessionPage];
  const currentSessionOriginalIndex = sessions.length - 1 - sessionPage;

  const kbims = currentSession.prediction?.kbims;
  const pps = currentSession.prediction?.pps;
  const pairCount = getPairCount(currentSession.prediction);
  const pairs = Array.from({ length: pairCount }, (_, i) => ({
    kbims: kbims!.candidates[i],
    pps: pps!.candidates[i],
  }));

  return (
    <Card>
      <CardHeader><CardTitle>{t.predict.results}</CardTitle></CardHeader>
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
                    aria-label={t.predict.prevSession}
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
                    {t.predict.session(currentSessionOriginalIndex + 1, sessions.length)}
                  </span>
                </PaginationItem>
                <PaginationItem>
                  <PaginationLink
                    onClick={(e) => {
                      e.preventDefault();
                      if (sessionPage < sessions.length - 1) setSessionPage((p) => p + 1);
                    }}
                    aria-label={t.predict.nextSession}
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
                  {t.predict.sessionLabel(currentSessionOriginalIndex + 1)}
                </h3>
                <span className="text-muted-foreground text-xs">
                  {new Date(currentSession.predicted_at).toLocaleString('ko-KR')}
                </span>
              </div>

              <div className="space-y-2">
                {pairs.map((pair, pairIdx) => (
                  <button
                    key={pairIdx}
                    type="button"
                    onClick={() => onSelectCandidate(currentSessionOriginalIndex, pairIdx)}
                    className={cn(
                      'w-full rounded-lg border p-3 text-left transition-colors',
                      currentSession.selectedIndex === pairIdx
                        ? 'border-primary bg-primary/5'
                        : 'hover:bg-muted/50',
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground text-xs font-medium">
                          {t.predict.rank(pairIdx + 1)}
                        </span>
                        <Badge className={confidenceBadgeClass(pair.kbims.llm_confidence)}>
                          KBIMS {(pair.kbims.llm_confidence * 100).toFixed(0)}%
                        </Badge>
                        <Badge className={confidenceBadgeClass(pair.pps.llm_confidence)}>
                          PPS {(pair.pps.llm_confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      {currentSession.selectedIndex === pairIdx && (
                        <span className="text-primary text-xs font-medium">{t.predict.selected}</span>
                      )}
                    </div>
                    <div className="mt-2 space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground text-xs w-14">{t.object.partCode}</span>
                        <span className="text-sm font-semibold">{pair.kbims.code || '-'}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground text-xs w-14">{t.object.ppsCode}</span>
                        <span className="text-sm font-semibold">{pair.pps.code || '-'}</span>
                      </div>
                    </div>
                    {(pair.kbims.reasoning || pair.pps.reasoning) && (
                      <p className="text-muted-foreground mt-2 text-xs line-clamp-2">
                        {pair.kbims.reasoning ?? pair.pps.reasoning}
                      </p>
                    )}
                  </button>
                ))}

                {/* User input card */}
                <button
                  type="button"
                  onClick={() => onSelectCandidate(currentSessionOriginalIndex, pairCount)}
                  className={cn(
                    'w-full rounded-lg border p-3 text-left transition-colors',
                    currentSession.selectedIndex === pairCount
                      ? 'border-primary bg-primary/5'
                      : 'hover:bg-muted/50',
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-muted-foreground text-xs font-medium">{t.input.userInput}</span>
                      <Badge className="bg-gray-100 text-gray-800">{t.input.manualInput}</Badge>
                    </div>
                    {currentSession.selectedIndex === pairCount && (
                      <span className="text-primary text-xs font-medium">{t.predict.selected}</span>
                    )}
                  </div>
                  <div className="mt-2 space-y-2" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">{t.object.partCode}</span>
                      <Input
                        value={currentSession.userCandidate?.kbims_code ?? ''}
                        onChange={(e) =>
                          onUserCandidateChange(currentSessionOriginalIndex, {
                            kbims_code: e.target.value,
                            pps_code: currentSession.userCandidate?.pps_code ?? '',
                            reasoning: currentSession.userCandidate?.reasoning,
                          })
                        }
                        placeholder={t.input.partCodePlaceholder}
                        className="h-7 text-sm"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">{t.object.ppsCode}</span>
                      <Input
                        value={currentSession.userCandidate?.pps_code ?? ''}
                        onChange={(e) =>
                          onUserCandidateChange(currentSessionOriginalIndex, {
                            kbims_code: currentSession.userCandidate?.kbims_code ?? '',
                            pps_code: e.target.value,
                            reasoning: currentSession.userCandidate?.reasoning,
                          })
                        }
                        placeholder={t.input.ppsCodePlaceholder}
                        className="h-7 text-sm"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-xs w-14 shrink-0">{t.input.description}</span>
                      <Input
                        value={currentSession.userCandidate?.reasoning ?? ''}
                        onChange={(e) =>
                          onUserCandidateChange(currentSessionOriginalIndex, {
                            kbims_code: currentSession.userCandidate?.kbims_code ?? '',
                            pps_code: currentSession.userCandidate?.pps_code ?? '',
                            reasoning: e.target.value,
                          })
                        }
                        placeholder={t.input.descriptionPlaceholder}
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
              {isPredicting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t.predict.rePredict}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
