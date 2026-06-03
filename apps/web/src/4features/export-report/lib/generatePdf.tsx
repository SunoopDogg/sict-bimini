import { pdf } from '@react-pdf/renderer';

import type { ReportData } from '../model/types';
import { ReportDocument } from './ReportDocument';

export async function downloadReportPdf(
  data: ReportData,
  fileBase: string,
): Promise<void> {
  const blob = await pdf(<ReportDocument data={data} />).toBlob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `report_${fileBase}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
