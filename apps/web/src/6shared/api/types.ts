export interface APIResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

export interface MetaResponse {
  llm_model: string;
  embedding_model: string;
}
