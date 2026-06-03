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
  section: { marginBottom: 10 },
  candRow: { flexDirection: 'row', paddingVertical: 1 },
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

function DetailSection({ row, index }: { row: ReportObjectRow; index: number }) {
  const o = row.object;
  const session = row.session;
  return (
    <View style={s.section} wrap={false}>
      <Text style={s.h2}>
        {index + 1}. {o.family_name || o.name || '객체'} / {o.type}
      </Text>
      <Text>
        ifc_type={o.ifc_type} category={o.category} family={o.family} type_id=
        {o.type_id}
      </Text>
      {!session ? (
        <Text style={s.muted}>미예측</Text>
      ) : (
        (['kbims', 'pps'] as const).map((tgt) => {
          const resp = session.prediction[tgt];
          return (
            <View key={tgt}>
              <Text style={{ fontWeight: 'bold', marginTop: 4 }}>
                {tgt.toUpperCase()} [{resp.mode}] pool={resp.pool_size} k=
                {resp.retrieved_k}
              </Text>
              {resp.candidates.map((c, ci) => (
                <View key={ci} style={s.candRow}>
                  <Text style={{ width: '14%' }}>{c.code}</Text>
                  <Text style={{ width: '14%' }}>
                    conf {c.llm_confidence.toFixed(2)}
                  </Text>
                  <Text style={{ width: '14%' }}>
                    ret{' '}
                    {c.retrieval_score === null
                      ? '—'
                      : c.retrieval_score.toFixed(2)}
                  </Text>
                  <Text style={{ width: '14%' }}>{c.source}</Text>
                  <Text style={{ width: '44%' }}>{c.reasoning ?? ''}</Text>
                </View>
              ))}
            </View>
          );
        })
      )}
      {session && (
        <Text style={{ marginTop: 3, color: '#555' }}>
          최종 선택: KBIMS {row.finalKbims ?? '—'} / PPS {row.finalPps ?? '—'}
          {'   '}({session.predicted_at})
        </Text>
      )}
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

        <Text style={[s.h2, { marginTop: 16 }]} break>
          객체별 상세
        </Text>
        {data.rows.map((r, i) => (
          <DetailSection key={i} row={r} index={i} />
        ))}
      </Page>
    </Document>
  );
}
