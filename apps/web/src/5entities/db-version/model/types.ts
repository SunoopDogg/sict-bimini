export interface DbVersion {
  name: string;
  points: number;
}

export interface DbVersionListResponse {
  versions: DbVersion[];
}
