export interface BIMObject {
  name?: string;       // xlsx display only — never sent to API
  ifc_type: string;    // was object_type
  category: string;
  family_name: string;
  family: string;
  type: string;
  type_id: string;
  kbims_code: string;
  pps_code: string;
}

export const EMPTY_BIM_OBJECT: BIMObject = {
  ifc_type: '',
  category: '',
  family_name: '',
  family: '',
  type: '',
  type_id: '',
  kbims_code: '',
  pps_code: '',
};
