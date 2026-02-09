export interface BIMObjectInput {
  name: string;
  object_type: string;
  category: string;
  family_name: string;
  family: string;
  type: string;
  type_id: string;
  pps_code: string;
  kbims_code: string;
}

export const EMPTY_BIM_OBJECT: BIMObjectInput = {
  name: '',
  object_type: '',
  category: '',
  family_name: '',
  family: '',
  type: '',
  type_id: '',
  pps_code: '',
  kbims_code: '',
};
