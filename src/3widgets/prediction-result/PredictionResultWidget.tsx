'use client';

import { useState } from 'react';
import { useActionState } from 'react';
import { useFormStatus } from 'react-dom';

import { type FeedbackState, feedbackAction } from '@/app/predict/actions';
import type { BIMObjectInput } from '@/src/5entities/bim-object';
import type {
  Prediction,
  PredictionResponse,
} from '@/src/5entities/prediction';
import { cn } from '@/src/6shared/lib/cn';
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from '@/src/6shared/ui/primitive/alert';
import { Badge } from '@/src/6shared/ui/primitive/badge';
import { Button } from '@/src/6shared/ui/primitive/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/src/6shared/ui/primitive/card';
import { Input } from '@/src/6shared/ui/primitive/input';
import { Label } from '@/src/6shared/ui/primitive/label';

interface PredictionResultWidgetProps {
  response: PredictionResponse;
  bimInput: BIMObjectInput;
  onReset?: () => void;
}

function SubmitFeedbackButton() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" disabled={pending} size="sm">
      {pending ? '제출 중...' : '피드백 제출'}
    </Button>
  );
}

function PredictionCard({
  prediction,
  onSelect,
  isSelected,
}: {
  prediction: Prediction;
  onSelect: () => void;
  isSelected: boolean;
}) {
  const confidenceColor =
    prediction.confidence >= 0.7
      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      : prediction.confidence >= 0.4
        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-md',
        isSelected && 'ring-primary ring-2',
      )}
      onClick={onSelect}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline">#{prediction.rank}</Badge>
            <CardTitle className="text-lg">{prediction.code}</CardTitle>
          </div>
          <Badge className={confidenceColor}>
            {(prediction.confidence * 100).toFixed(0)}%
          </Badge>
        </div>
        <CardDescription>{prediction.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground text-sm">{prediction.reasoning}</p>
      </CardContent>
    </Card>
  );
}

export function PredictionResultWidget({
  response,
  bimInput,
  onReset,
}: PredictionResultWidgetProps) {
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackState, feedbackFormAction] = useActionState(
    feedbackAction,
    null,
  );

  const handleSelect = (code: string) => {
    setSelectedCode(code);
    setShowFeedback(false);
  };

  const handleReject = () => {
    setSelectedCode(null);
    setShowFeedback(true);
  };

  if (feedbackState?.success) {
    return (
      <Alert>
        <AlertTitle>피드백 완료</AlertTitle>
        <AlertDescription>
          {feedbackState.message ||
            '피드백이 성공적으로 제출되었습니다. 벡터 DB에 추가되었습니다.'}
        </AlertDescription>
        <Button onClick={onReset} className="mt-4" variant="outline">
          새로운 예측 시작
        </Button>
      </Alert>
    );
  }

  if (selectedCode) {
    return (
      <Alert>
        <AlertTitle>선택 완료</AlertTitle>
        <AlertDescription>
          부위코드 <strong>{selectedCode}</strong>가 선택되었습니다.
        </AlertDescription>
        <Button onClick={onReset} className="mt-4" variant="outline">
          새로운 예측 시작
        </Button>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">예측 결과 (Top 3)</h2>
        <Button variant="outline" size="sm" onClick={handleReject}>
          맞는 코드가 없어요
        </Button>
      </div>

      <div className="grid gap-4">
        {response.predictions.map((prediction) => (
          <PredictionCard
            key={prediction.code}
            prediction={prediction}
            onSelect={() => handleSelect(prediction.code)}
            isSelected={selectedCode === prediction.code}
          />
        ))}
      </div>

      {showFeedback && (
        <Card className="border-dashed">
          <CardHeader>
            <CardTitle className="text-lg">올바른 부위코드 입력</CardTitle>
            <CardDescription>
              정확한 KBIMS 부위코드를 입력해 주세요. 입력된 데이터는 벡터 DB에
              추가됩니다.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form action={feedbackFormAction} className="space-y-4">
              <input
                type="hidden"
                name="request_id"
                value={response.request_id}
              />
              <input
                type="hidden"
                name="bim_data"
                value={JSON.stringify(bimInput)}
              />
              <div className="space-y-2">
                <Label htmlFor="correct_code">올바른 부위코드</Label>
                <Input
                  id="correct_code"
                  name="correct_code"
                  placeholder="E23"
                  required
                />
              </div>
              {feedbackState?.error && (
                <Alert variant="destructive">
                  <AlertDescription>{feedbackState.error}</AlertDescription>
                </Alert>
              )}
              <div className="flex gap-2">
                <SubmitFeedbackButton />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowFeedback(false)}
                >
                  취소
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
