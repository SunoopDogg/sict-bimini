export type {
  PredictionMode,
  PredictionCandidate,
  PredictionResponse,
  CombinedPredictionResponse,
  PredictionSession,
  UserCandidate,
  UserSelection,
  SelectionFileInfo,
  SelectionFileData,
  BatchItemResult,
  BatchPredictResult,
} from './model/types';

export { getPairCount } from './lib/getPairCount';
export {
  classifyPredictionMatch,
  type PredictionMatchStatus,
} from './lib/classifyPredictionMatch';
export { findSessionForVersion } from './lib/findSessionForVersion';
export { sessionMatchesVersion } from './lib/sessionMatchesVersion';
export {
  getSelectedPrediction,
  type SelectedPrediction,
} from './lib/getSelectedPrediction';
export {
  EMPTY_COMBINED_PREDICTION_RESPONSE,
  buildSelectionSessionMap,
} from './lib/predictionSessions';
