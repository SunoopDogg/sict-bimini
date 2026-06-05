export type PredictionMatchStatus =
  | 'unpredicted' // 예측 코드 없음
  | 'no-truth' // 예측은 있으나 정답(GT) 없음 → 판정 불가
  | 'match' // 예측 == 정답
  | 'mismatch'; // 예측 != 정답

/**
 * Classify a predicted code against ground truth — the single source of truth
 * for the predicted-vs-GT rule. Consumers map the status to their own
 * presentation: the object-list panel → icons, the PDF report → 'O'/'X'/code,
 * the report stats → judged correct/incorrect counts.
 */
export function classifyPredictionMatch(
  predicted: string | null,
  actual: string,
): PredictionMatchStatus {
  if (!predicted) return 'unpredicted';
  if (!actual) return 'no-truth';
  return predicted === actual ? 'match' : 'mismatch';
}
