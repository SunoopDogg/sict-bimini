'use client';

import { useState } from 'react';

import { BatchPredictionResultWidget } from '@/src/3widgets/batch-prediction-result';
import {
  type BatchPredictionResult,
  PredictForm,
} from '@/src/4features/predict-code';
import { Alert, AlertDescription } from '@/src/6shared/ui/primitive/alert';

export default function PredictPage() {
  const [result, setResult] = useState<BatchPredictionResult | null>(null);

  const handleResult = (state: BatchPredictionResult) => {
    setResult(state);
  };

  const handleReset = () => {
    setResult(null);
  };

  return (
    <main className="container mx-auto max-w-4xl px-4 py-8">
      <h1 className="mb-8 text-center text-3xl font-bold">
        KBIMS 부위코드 예측
      </h1>

      {!result?.success ? (
        <>
          <PredictForm onResult={handleResult} />
          {result?.error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{result.error}</AlertDescription>
            </Alert>
          )}
        </>
      ) : (
        result.data && (
          <BatchPredictionResultWidget
            result={result.data}
            onReset={handleReset}
          />
        )
      )}
    </main>
  );
}
