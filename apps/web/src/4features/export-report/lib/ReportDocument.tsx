import {
  Document,
  Font,
  Page,
  StyleSheet,
  Text,
  View,
} from '@react-pdf/renderer';

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

// 요약표 열 너비(합 100)
const COLS = [22, 14, 14, 12, 12, 10, 10, 6];

function fmtConf(v: number | null): string {
  return v === null ? '—' : v.toFixed(2);
}

function SummaryTable({ rows }: { rows: ReportObjectRow[] }) {
  const head = [
    '식별',
    'ifc_type',
    'category',
    'KBIMS',
    'PPS',
    'KBIMS신뢰',
    'PPS신뢰',
    '모드',
  ];
  return (
    <View>
      <View style={[s.tableRow, s.th]}>
        {head.map((h, i) => (
          <Text key={h} style={[s.cell, { width: `${COLS[i]}%` }]}>
            {h}
          </Text>
        ))}
      </View>
      {rows.map((r, i) => {
        const ident =
          `${r.object.family_name || r.object.name || ''} ${r.object.type}`.trim();
        const mode = r.session ? r.session.prediction.kbims.mode : '';
        const vals = [
          ident || '—',
          r.object.ifc_type,
          r.object.category,
          r.finalKbims ?? '—',
          r.finalPps ?? '—',
          fmtConf(r.kbimsConfidence),
          fmtConf(r.ppsConfidence),
          mode || '—',
        ];
        return (
          <View key={i} style={s.tableRow}>
            {vals.map((v, j) => (
              <Text
                key={j}
                style={[
                  s.cell,
                  { width: `${COLS[j]}%` },
                  !r.session ? s.muted : {},
                ]}
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
    ['객체수', `${m.predictedCount} / ${m.objectCount} 예측완료`],
  ];
  return (
    <Document>
      <Page size="A4" style={s.page}>
        <Text style={s.h1}>예측 결과 보고서</Text>
        <View style={{ marginBottom: 8 }}>
          {metaRows.map(([k, v]) => (
            <View key={k} style={s.metaRow}>
              <Text style={s.metaKey}>{k}</Text>
              <Text>{v}</Text>
            </View>
          ))}
        </View>

        <Text style={s.h2}>요약</Text>
        <SummaryTable rows={data.rows} />
      </Page>
    </Document>
  );
}
