export interface APIResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}
