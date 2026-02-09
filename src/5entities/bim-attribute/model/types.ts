export interface BIMAttributeItem {
  ifc_type: string;
  category: string;
  family_name: string;
  kbims_code: string;
  pps_code: string;
  family: string;
  type: string;
  type_id: string;
}

export interface BIMAttributeListResponse {
  items: BIMAttributeItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}
