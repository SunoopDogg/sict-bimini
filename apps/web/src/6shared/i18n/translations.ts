export type Locale = 'ko' | 'en';

const ko = {
  pageTitle: 'KBIMS · PPS 코드 예측',
  themeToggle: '테마 전환',
  localeToggle: '언어 전환',

  server: {
    online: '서버 온라인',
    degraded: '서버 불안정',
    offline: '서버 오프라인',
    version: '버전',
    connected: '연결됨',
    notConnected: '연결 안됨',
    cannotConnect: '서버에 연결할 수 없습니다.',
  },

  file: {
    sectionTitle: '파일',
    noFiles: '업로드된 파일이 없습니다.',
    uploadedFiles: (n: number) => `업로드된 파일 (${n}개)`,
    selectFilePrompt: '파일을 선택하면 객체 리스트가 표시됩니다.',
    noObjects: '객체가 없습니다.',
    objectList: '객체 리스트',
    objects: (n: number) => `${n}개 객체`,
    prevPage: '이전 페이지',
    nextPage: '다음 페이지',
  },

  upload: {
    uploading: '업로드 및 변환 중...',
    pleaseWait: '잠시만 기다려주세요',
    done: '변환 완료!',
    dragOrClick: 'xlsx 파일을 드래그하거나',
    selectFile: '파일 선택',
    xlsxOnly: '.xlsx 형식만 지원됩니다',
  },

  version: {
    title: 'DB 버전',
    none: '버전 없음',
    items: (n: number) => `${n}개`,
  },

  createVersion: {
    title: 'DB 버전 생성',
    dbName: 'DB 이름',
    dbNamePlaceholder: '새 버전 이름 (영문/숫자/_/-)',
    baseDb: '기반 DB',
    baseNone: '(없음 · 빈 DB)',
    threshold: '신뢰도 임계값',
    addByConfidence: (n: number) => `신뢰도 ≥${n}% 객체 추가`,
    create: '버전 생성',
    creating: '생성 중…',
    targetList: (n: number) => `업데이트 대상 (${n})`,
    colName: '객체 이름',
    colKbims: '부위코드',
    colPps: 'PPS 코드',
    colSource: '출처',
    colRemove: '삭제',
    sourceUser: '사용자입력',
    sourceConfidence: '신뢰도',
    sourceConfidenceThreshold: (n: number) => `신뢰도 ≥${n}%`,
    empty: '업데이트 대상이 없습니다. 사용자 입력 또는 신뢰도 추가 버튼을 사용하세요.',
    noObjects: '파일을 선택하고 예측을 실행하세요.',
    success: (v: string, added: number, total: number) =>
      `'${v}' 생성 완료 — 추가 ${added}, 전체 ${total}`,
  },

  predict: {
    results: '예측 결과',
    predict: '예측하기',
    rePredict: '다시 예측하기',
    selectObjectPrompt: '객체를 선택하면 예측 결과가 표시됩니다.',
    batchPredict: (n: number) => `선택 예측 (${n}개)`,
    session: (current: number, total: number) => `예측 #${current} / ${total}`,
    sessionLabel: (n: number) => `예측 #${n}`,
    rank: (n: number) => `${n}순위`,
    selected: '선택됨',
    prevSession: '이전 예측',
    nextSession: '다음 예측',
    failed: '예측에 실패했습니다.',
  },

  report: {
    title: '보고서',
    fileLabel: '파일',
    export: '보고서 추출',
    predicting: (done: number, total: number) => `예측 중… ${done}/${total}`,
    cancel: '중단',
    cancelling: '중단 중…',
    generating: '생성 중…',
    noObjects: '객체 없음',
    failed: '보고서 생성 실패',
  },

  object: {
    info: '객체 정보',
    name: '이름',
    type: '유형',
    category: '카테고리',
    family: '패밀리',
    partCode: '부위코드',
    ppsCode: 'PPS 코드',
    colName: '객체 이름',
    colPartCode: '부위코드',
    colPpsCode: 'PPS 코드',
    colPredPartCode: '예측(부위)',
    colPredPpsCode: '예측(PPS)',
  },

  input: {
    userInput: '사용자 입력',
    manualInput: '직접 입력',
    description: '설명',
    partCodePlaceholder: '부위코드 입력',
    ppsCodePlaceholder: 'PPS 코드 입력',
    descriptionPlaceholder: '설명 입력',
  },

  bimAttr: {
    trigger: '속성 데이터',
    title: 'BIM 속성 데이터 (CSV)',
    subtitle: '벡터 데이터베이스에 저장된 BIM 속성 목록입니다.',
    noData: '데이터가 없습니다.',
    total: (n: number) => `총 ${n.toLocaleString()}건`,
    colIfcType: 'IFC 타입',
    colCategory: '분류',
    colFamilyName: '패밀리명',
    colKbimsCode: 'KBIMS 코드',
    colPpsCode: 'PPS 코드',
    colFamily: '패밀리',
    colType: '유형',
    colTypeId: '유형ID',
    loadFailed: '데이터를 불러오는데 실패했습니다.',
    loadError: '데이터를 불러오는 중 오류가 발생했습니다.',
  },

  userSel: {
    title: '사용자 선택',
    noFiles: '사용자 선택 파일이 없습니다.',
    items: (n: number) => `${n}개 항목`,
  },

  errors: {
    loadFilesFailed: '파일 목록을 불러올 수 없습니다.',
    uploadFailed: '파일 업로드에 실패했습니다.',
    loadObjectsFailed: '객체 데이터를 불러올 수 없습니다.',
    loadSelectionsFailed: '사용자 선택을 불러올 수 없습니다.',
  },

  settings: {
    language: '언어',
    theme: '테마',
  },
};

const en: typeof ko = {
  pageTitle: 'KBIMS · PPS Code Prediction',
  themeToggle: 'Toggle theme',
  localeToggle: 'Toggle language',

  server: {
    online: 'Server Online',
    degraded: 'Server Unstable',
    offline: 'Server Offline',
    version: 'Version',
    connected: 'Connected',
    notConnected: 'Not Connected',
    cannotConnect: 'Cannot connect to server.',
  },

  file: {
    sectionTitle: 'Files',
    noFiles: 'No uploaded files.',
    uploadedFiles: (n: number) => `Uploaded files (${n})`,
    selectFilePrompt: 'Select a file to view objects.',
    noObjects: 'No objects.',
    objectList: 'Object List',
    objects: (n: number) => `${n} objects`,
    prevPage: 'Previous page',
    nextPage: 'Next page',
  },

  upload: {
    uploading: 'Uploading and converting...',
    pleaseWait: 'Please wait',
    done: 'Conversion complete!',
    dragOrClick: 'Drag an xlsx file or',
    selectFile: 'Select file',
    xlsxOnly: 'Only .xlsx format supported',
  },

  version: {
    title: 'DB Version',
    none: 'No versions',
    items: (n: number) => `${n}`,
  },

  createVersion: {
    title: 'Create DB Version',
    dbName: 'DB name',
    dbNamePlaceholder: 'New version name (A–Z, 0–9, _, -)',
    baseDb: 'Base DB',
    baseNone: '(none · empty DB)',
    threshold: 'Confidence threshold',
    addByConfidence: (n: number) => `Add objects ≥${n}%`,
    create: 'Create version',
    creating: 'Creating…',
    targetList: (n: number) => `Update targets (${n})`,
    colName: 'Object name',
    colKbims: 'Part code',
    colPps: 'PPS code',
    colSource: 'Source',
    colRemove: 'Remove',
    sourceUser: 'User input',
    sourceConfidence: 'Confidence',
    sourceConfidenceThreshold: (n: number) => `Confidence ≥${n}%`,
    empty: 'No update targets. Use user input or the confidence-add button.',
    noObjects: 'Select a file and run predictions.',
    success: (v: string, added: number, total: number) =>
      `Created '${v}' — added ${added}, total ${total}`,
  },

  predict: {
    results: 'Prediction Results',
    predict: 'Predict',
    rePredict: 'Re-Predict',
    selectObjectPrompt: 'Select an object to view prediction results.',
    batchPredict: (n: number) => `Predict Selected (${n})`,
    session: (current: number, total: number) => `Prediction #${current} / ${total}`,
    sessionLabel: (n: number) => `Prediction #${n}`,
    rank: (n: number) => `Rank ${n}`,
    selected: 'Selected',
    prevSession: 'Previous prediction',
    nextSession: 'Next prediction',
    failed: 'Prediction failed.',
  },

  report: {
    title: 'Report',
    fileLabel: 'File',
    export: 'Export report',
    predicting: (done: number, total: number) => `Predicting… ${done}/${total}`,
    cancel: 'Cancel',
    cancelling: 'Cancelling…',
    generating: 'Generating…',
    noObjects: 'No objects',
    failed: 'Report generation failed',
  },

  object: {
    info: 'Object Info',
    name: 'Name',
    type: 'Type',
    category: 'Category',
    family: 'Family',
    partCode: 'Part Code',
    ppsCode: 'PPS Code',
    colName: 'Object Name',
    colPartCode: 'Part Code',
    colPpsCode: 'PPS Code',
    colPredPartCode: 'Pred (Part)',
    colPredPpsCode: 'Pred (PPS)',
  },

  input: {
    userInput: 'User Input',
    manualInput: 'Manual Input',
    description: 'Description',
    partCodePlaceholder: 'Enter part code',
    ppsCodePlaceholder: 'Enter PPS code',
    descriptionPlaceholder: 'Enter description',
  },

  bimAttr: {
    trigger: 'Attribute Data',
    title: 'BIM Attribute Data (CSV)',
    subtitle: 'BIM attribute list stored in the vector database.',
    noData: 'No data.',
    total: (n: number) => `Total ${n.toLocaleString()} records`,
    colIfcType: 'IFC Type',
    colCategory: 'Category',
    colFamilyName: 'Family Name',
    colKbimsCode: 'KBIMS Code',
    colPpsCode: 'PPS Code',
    colFamily: 'Family',
    colType: 'Type',
    colTypeId: 'Type ID',
    loadFailed: 'Failed to load data.',
    loadError: 'An error occurred while loading data.',
  },

  userSel: {
    title: 'User Selections',
    noFiles: 'No user selection files.',
    items: (n: number) => `${n} items`,
  },

  errors: {
    loadFilesFailed: 'Failed to load file list.',
    uploadFailed: 'File upload failed.',
    loadObjectsFailed: 'Failed to load object data.',
    loadSelectionsFailed: 'Failed to load user selections.',
  },

  settings: {
    language: 'Language',
    theme: 'Theme',
  },
};

export const translations = { ko, en } satisfies Record<Locale, typeof ko>;
export type Translations = typeof ko;
