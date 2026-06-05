export interface DbVersion {
  name: string;
  points: number;
}

export interface DbVersionListResponse {
  versions: DbVersion[];
}

export interface VersionCreateItem {
  ifc_type: string;
  category: string;
  family_name: string;
  family: string;
  type: string;
  type_id: string;
  kbims_code: string;
  pps_code: string;
}

export interface CreateVersionInput {
  name: string;
  base: string | null;
  items: VersionCreateItem[];
}

export interface CreateVersionResult {
  version: string;
  copied: number;
  added: number;
  total: number;
}
