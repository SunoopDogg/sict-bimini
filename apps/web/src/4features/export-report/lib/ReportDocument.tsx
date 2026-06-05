import {
  Document,
  Font,
  Page,
  StyleSheet,
  Text,
  View,
} from '@react-pdf/renderer';

import { classifyPredictionMatch } from '@/5entities/prediction';

import type { ReportData, ReportObjectRow } from '../model/types';

Font.register({
  family: 'NotoSansKR',
  fonts: [
    { src: '/fonts/NotoSansKR-Regular.ttf' },
    { src: '/fonts/NotoSansKR-Bold.ttf', fontWeight: 'bold' },
  ],
});

const s = StyleSheet.create({
  page: { fontFamily: 'NotoSansKR', fontSize: 8, padding: 28, color: '#111' },
  h1: { fontSize: 14, fontWeight: 'bold', marginBottom: 6 },
  h2: { fontSize: 11, fontWeight: 'bold', marginTop: 12, marginBottom: 4 },
  metaRow: { flexDirection: 'row', marginBottom: 2 },
  metaKey: { width: 90, color: '#555' },
  tableRow: {
    flexDirection: 'row',
    borderBottom: '1px solid #ddd',
    paddingVertical: 2,
  },
  th: { fontWeight: 'bold', backgroundColor: '#f2f2f2' },
  cell: { paddingHorizontal: 3 },
  muted: { color: '#999' },
});

// 요약표 열 너비(합 100) — web 객체 리스트 패널과 동일 구성
const COLS = [6, 30, 16, 16, 16, 16];
// 너비 스타일을 한 번만 만들어 셀마다 재사용(매 셀 객체 생성 방지).
const CELL_W = COLS.map((w) => ({ width: `${w}%` }));
// 미예측 행이 아닐 때 쓰는 빈 스타일(공유 참조 — 매 셀 {} 생성 방지).
const NO_STYLE = {};

function fmtConf(v: number | null): string {
  return v === null ? '—' : v.toFixed(2);
}

function fmtRatio(n: number, d: number): string {
  return d === 0 ? '—' : `${n} / ${d} (${Math.round((n / d) * 100)}%)`;
}

// 메타·통계 공용 키/값 행 렌더. 두 섹션이 동일 레이아웃을 공유.
function KeyValueRows({ rows }: { rows: [string, string][] }) {
  return (
    <>
      {rows.map(([k, v]) => (
        <View key={k} style={s.metaRow}>
          <Text style={s.metaKey}>{k}</Text>
          <Text>{v}</Text>
        </View>
      ))}
    </>
  );
}

function StatsSection({ data }: { data: ReportData }) {
  const st = data.stats;
  const statRows: [string, string][] = [
    ['예측 완료율', fmtRatio(st.predictedCount, st.objectCount)],
    [
      'KBIMS 정확도',
      st.kbimsJudgeable
        ? fmtRatio(st.kbimsCorrect, st.kbimsJudgeable)
        : '— (정답 없음)',
    ],
    [
      'PPS 정확도',
      st.ppsJudgeable
        ? fmtRatio(st.ppsCorrect, st.ppsJudgeable)
        : '— (정답 없음)',
    ],
    [
      '평균 신뢰도',
      `KBIMS ${fmtConf(st.avgKbimsConfidence)}  ·  PPS ${fmtConf(st.avgPpsConfidence)}`,
    ],
  ];
  return (
    <View style={{ marginBottom: 4 }}>
      <KeyValueRows rows={statRows} />
    </View>
  );
}

// 예측 셀: web 객체 리스트와 동일 규칙(classifyPredictionMatch 공유).
// 정답 보유 → O/X, 정답 없음 → 예측 코드, 미예측 → '—'.
function matchCell(predicted: string | null, actual: string): string {
  switch (classifyPredictionMatch(predicted, actual)) {
    case 'match':
      return 'O';
    case 'mismatch':
      return 'X';
    case 'no-truth':
      return predicted ?? '—';
    default:
      return '—';
  }
}

function SummaryTable({ rows }: { rows: ReportObjectRow[] }) {
  const head = [
    '#',
    '객체 이름',
    '부위코드',
    '예측(부위)',
    'PPS 코드',
    '예측(PPS)',
  ];
  return (
    <View>
      <View style={[s.tableRow, s.th]}>
        {head.map((h, i) => (
          <Text key={h} style={[s.cell, CELL_W[i]]}>
            {h}
          </Text>
        ))}
      </View>
      {rows.map((r, i) => {
        const vals = [
          String(i + 1),
          r.object.name || '-',
          r.object.kbims_code || '-',
          matchCell(r.finalKbims, r.object.kbims_code),
          r.object.pps_code || '-',
          matchCell(r.finalPps, r.object.pps_code),
        ];
        return (
          <View key={i} style={s.tableRow}>
            {vals.map((v, j) => (
              <Text
                key={j}
                style={[s.cell, CELL_W[j], r.session ? NO_STYLE : s.muted]}
              >
                {v}
              </Text>
            ))}
          </View>
        );
      })}
    </View>
  );
}

export function ReportDocument({ data }: { data: ReportData }) {
  const m = data.meta;
  const metaRows: [string, string][] = [
    ['DB 버전', m.version ?? '—'],
    ['LLM', m.llmModel],
    ['임베딩', m.embeddingModel],
    ['소스 파일', m.fileName ?? '—'],
    ['생성', m.generatedAt],
  ];
  return (
    <Document>
      <Page size="A4" style={s.page}>
        <Text style={s.h1}>예측 결과 보고서</Text>
        <View style={{ marginBottom: 8 }}>
          <KeyValueRows rows={metaRows} />
        </View>

        <Text style={s.h2}>통계</Text>
        <StatsSection data={data} />

        <Text style={s.h2}>요약</Text>
        <SummaryTable rows={data.rows} />
      </Page>
    </Document>
  );
}
