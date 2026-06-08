'use client';

import { useEffect, useState, useTransition } from 'react';

import { usePredictionSessions } from '@/2pages/predict/hooks/usePredictionSessions';
import { useUserSelections } from '@/2pages/predict/hooks/useUserSelections';
import { ObjectPredictionPanel } from '@/3widgets/object-prediction-panel';
import { BimAttributeTableModal } from '@/4features/bim-attribute-viewer';
import { useVersions, VersionSelect } from '@/4features/select-db-version';
import { CreateVersionPanel } from '@/4features/create-db-version';
import { ServerStatusBadge } from '@/4features/server-status';
import {
  FileListSelect,
  FileUploadZone,
  ObjectListPanel,
  type UploadStatus,
  listXlsxFilesAction,
  readJsonFileAction,
  uploadAndConvertXlsxAction,
} from '@/4features/manage-file';
import { loadPredictionsAction } from '@/4features/predict-code';
import { ExportReportButton } from '@/4features/export-report';
import type { BIMObject } from '@/5entities/bim-object';
import type { PredictionSession } from '@/5entities/prediction';
import type { XlsxFileInfo } from '@/5entities/xlsx-file';
import { batchPredictCode, predictSingleCode } from '@/6shared/api';
import { cn } from '@/6shared/lib/cn';
import { useLocale } from '@/6shared/i18n';
import { SettingsDropdown } from '@/6shared/ui/SettingsDropdown';
import { Alert, AlertDescription } from '@/6shared/ui/primitive/alert';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/6shared/ui/primitive/card';

type DataSource = { type: 'xlsx'; fileName: string } | null;

export default function PredictPage() {
  const [files, setFiles] = useState<XlsxFileInfo[]>([]);
  const [activeSource, setDataSource] = useState<DataSource>(null);
  const [objects, setObjects] = useState<BIMObject[]>([]);
  const [selectedObjectIndex, setSelectedObjectIndex] = useState<number | null>(
    null,
  );
  // Independent prediction source for the create-version panel — file chosen
  // separately from the 1height view, so the update list is fully decoupled.
  const [createSourceFile, setCreateSourceFile] = useState<string>();
  const [createObjects, setCreateObjects] = useState<BIMObject[]>([]);
  const [createPredictionMap, setCreatePredictionMap] = useState<
    Record<string, PredictionSession[]>
  >({});
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>('idle');
  const [isLoadingObjects, setIsLoadingObjects] = useState(false);
  const [error, setError] = useState<string>();
  const [isPredicting, startPrediction] = useTransition();
  const { t } = useLocale();
  const [predictingIndex, setPredictingIndex] = useState<number | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(
    new Set(),
  );
  const {
    versions,
    isLoading: versionsLoading,
    error: versionsError,
    refetch: refetchVersions,
  } = useVersions();
  const [pickedVersion, setPickedVersion] = useState<string>();
  // True while a report export (predict + PDF) runs — locks file/version
  // switching so an in-flight export can't clobber a freshly loaded file.
  const [isExporting, setIsExporting] = useState(false);
  // DB version whose contents are shown in the viewer modal (eye icon).
  const [viewVersion, setViewVersion] = useState<string | null>(null);
  // Visual + interaction lock applied to the file/version cards while exporting.
  const lockClass = cn(isExporting && 'pointer-events-none opacity-50');
  // Effective version: user's pick, else default to the first available.
  const selectedVersion = pickedVersion ?? versions[0]?.name;

  const selectedFile =
    activeSource?.type === 'xlsx' ? activeSource.fileName : undefined;

  const {
    refreshSelectionFiles,
    addToSelections,
    removeFromSelections,
    loadInitialSelections,
    syncSelectionsFromMap,
  } = useUserSelections(objects);

  const {
    predictionMap,
    setPredictionMap,
    appendSessions,
    toEntries,
    ensureAllPredicted,
    handleSelectCandidate,
    handleUserCandidateChange,
    toSession,
  } = usePredictionSessions({
    selectedFile,
    onSelectionSync: (objectIndex, sessionIndex, session, action) => {
      if (action === 'add') {
        addToSelections(objectIndex, sessionIndex, session);
      } else {
        removeFromSelections(objectIndex);
      }
    },
  });

  const refreshFileList = async () => {
    const response = await listXlsxFilesAction();
    if (response.success && response.data) {
      setFiles(response.data);
    } else {
      setError(response.error || t.errors.loadFilesFailed);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (isExporting) return;
    setUploadStatus('uploading');
    setError(undefined);

    const formData = new FormData();
    formData.append('file', file);

    const response = await uploadAndConvertXlsxAction(formData, true);

    if (response.success && response.data) {
      setUploadStatus('done');
      await refreshFileList();
      if (response.data.file?.name) {
        setDataSource({ type: 'xlsx', fileName: response.data.file.name });
      }
      setTimeout(() => setUploadStatus('idle'), 2000);
    } else {
      setUploadStatus('idle');
      setError(response.error || t.errors.uploadFailed);
    }
  };

  const handleSelectXlsxFile = (fileName: string) => {
    if (isExporting) return;
    setDataSource({ type: 'xlsx', fileName });
  };

  const handleBatchPredict = () => {
    setError(undefined);
    // Ascending order so this lines up with `objects.filter` (and thus the
    // backend's per-object result order). `Set` iterates in insertion order,
    // which differs when the user checks boxes out of order — zipping that
    // against the filtered objects would attach predictions to wrong rows.
    const selectedIndicesArray = objects
      .map((_, i) => i)
      .filter((i) => selectedIndices.has(i));
    const selectedObjects = selectedIndicesArray.map((i) => objects[i]);
    startPrediction(async () => {
      const response = await batchPredictCode(selectedObjects, 5, selectedVersion);

      if (response.success && response.data) {
        appendSessions(toEntries(response.data.results, selectedIndicesArray));
        setSelectedIndices(new Set());
      } else {
        setError(response.error || t.predict.failed);
      }
    });
  };

  const handleSinglePredict = (index: number) => {
    setError(undefined);
    setPredictingIndex(index);
    startPrediction(async () => {
      const response = await predictSingleCode(objects[index], 5, selectedVersion);

      if (response.success && response.data) {
        appendSessions([{ index, session: toSession(response.data) }]);
      } else {
        setError(response.error || t.predict.failed);
      }
      setPredictingIndex(null);
    });
  };

  // Initial load
  useEffect(() => {
    const loadInitial = async () => {
      await Promise.all([refreshFileList(), refreshSelectionFiles()]);
      await loadInitialSelections();
    };
    loadInitial();
  }, []);

  // Load objects when xlsx file is selected
  useEffect(() => {
    if (activeSource?.type !== 'xlsx') return;
    const fileName = activeSource.fileName;

    const loadObjects = async () => {
      setSelectedObjectIndex(null);
      setError(undefined);
      setSelectedIndices(new Set());
      setIsLoadingObjects(true);

      let loadedObjects: BIMObject[] = [];
      const response = await readJsonFileAction(fileName);

      if (response.success && response.data) {
        loadedObjects = response.data;
        setObjects(response.data);
      } else {
        setObjects([]);
        setError(response.error || t.errors.loadObjectsFailed);
      }

      let loadedMap: Record<string, PredictionSession[]> = {};
      const predResult = await loadPredictionsAction(fileName);
      if (predResult.success && predResult.data) {
        loadedMap = predResult.data;
        setPredictionMap(loadedMap);
      } else {
        setPredictionMap({});
      }

      syncSelectionsFromMap(loadedMap, loadedObjects);
      setIsLoadingObjects(false);
    };

    loadObjects();
  }, [activeSource?.type === 'xlsx' ? activeSource.fileName : null]);

  // Load the create-version source file independently of the 1height file.
  useEffect(() => {
    if (!createSourceFile) {
      setCreateObjects([]);
      setCreatePredictionMap({});
      return;
    }
    let active = true;
    const load = async () => {
      // Independent reads (separate data dirs) — fetch concurrently.
      const [objRes, predRes] = await Promise.all([
        readJsonFileAction(createSourceFile),
        loadPredictionsAction(createSourceFile),
      ]);
      if (!active) return;
      setCreateObjects(objRes.success && objRes.data ? objRes.data : []);
      setCreatePredictionMap(
        predRes.success && predRes.data ? predRes.data : {},
      );
    };
    load();
    return () => {
      active = false;
    };
  }, [createSourceFile]);

  return (
    <main className="container mx-auto px-4 py-6">
      <div className="relative mb-6 flex items-center justify-center">
        <div className="absolute left-0">
          <ServerStatusBadge />
        </div>
        <h1 className="text-3xl font-bold">{t.pageTitle}</h1>
        <div className="absolute right-0">
          <SettingsDropdown />
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 1height: 3-panel grid reserves a full viewport band */}
      <div className="grid min-h-screen grid-cols-[280px_1fr_380px] gap-4">
        {/* Panel 1: DB Version + File List + Report */}
        <div className="flex flex-col gap-4 min-h-0">
          <Card className={cn('flex flex-col', lockClass)}>
            <CardHeader>
              <CardTitle>{t.version.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-48 overflow-y-auto">
                <VersionSelect
                  versions={versions}
                  value={selectedVersion}
                  onChange={(v) => {
                    if (!isExporting) setPickedVersion(v);
                  }}
                  isLoading={versionsLoading}
                  error={versionsError}
                  onView={setViewVersion}
                />
              </div>
            </CardContent>
          </Card>
          <Card className={cn('flex flex-col', lockClass)}>
            <CardHeader>
              <CardTitle>{t.file.sectionTitle}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col gap-4 overflow-hidden">
              <FileUploadZone
                onUpload={handleFileUpload}
                uploadStatus={uploadStatus}
              />
              <div className="flex-1 overflow-y-auto">
                <FileListSelect
                  files={files}
                  selectedFile={selectedFile}
                  onSelect={handleSelectXlsxFile}
                />
              </div>
            </CardContent>
          </Card>
          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>{t.report.title}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-muted-foreground space-y-0.5 text-xs">
                <div className="truncate">DB: {selectedVersion ?? '—'}</div>
                <div className="truncate">
                  {t.report.fileLabel}: {activeSource?.fileName ?? '—'}
                </div>
              </div>
              <ExportReportButton
                className="w-full"
                objects={objects}
                predictionMap={predictionMap}
                onEnsureAllPredicted={(onProgress, shouldCancel) =>
                  ensureAllPredicted(
                    objects,
                    selectedVersion,
                    onProgress,
                    shouldCancel,
                  )
                }
                onBusyChange={setIsExporting}
                version={selectedVersion}
                fileName={activeSource?.fileName}
              />
            </CardContent>
          </Card>
        </div>

        {/* Panel 2: Object List */}
        <div className="min-h-0">
          <ObjectListPanel
            selectedFile={activeSource?.fileName}
            objects={objects}
            isLoading={isLoadingObjects}
            isPredicting={isPredicting}
            predictingIndex={predictingIndex}
            selectedIndices={selectedIndices}
            onSelectionChange={setSelectedIndices}
            predictionMap={predictionMap}
            focusedIndex={selectedObjectIndex}
            selectedVersion={selectedVersion}
            onPredict={handleBatchPredict}
            onRowClick={(_obj: BIMObject, index: number) => {
              setSelectedObjectIndex(index);
            }}
          />
        </div>

        {/* Panel 3: Prediction Results */}
        <div className="min-h-0">
          <ObjectPredictionPanel
            object={
              selectedObjectIndex !== null
                ? (objects[selectedObjectIndex] ?? null)
                : null
            }
            sessions={
              selectedObjectIndex !== null
                ? (predictionMap[selectedObjectIndex] ?? [])
                : []
            }
            selectedVersion={selectedVersion}
            isPredicting={
              predictingIndex === selectedObjectIndex || isPredicting
            }
            onPredict={() => {
              if (selectedObjectIndex !== null) {
                handleSinglePredict(selectedObjectIndex);
              }
            }}
            onSelectCandidate={(sessionIndex, candidateIndex) => {
              if (selectedObjectIndex !== null) {
                handleSelectCandidate(
                  selectedObjectIndex,
                  sessionIndex,
                  candidateIndex,
                );
              }
            }}
            onUserCandidateChange={(sessionIndex, candidate) => {
              if (selectedObjectIndex !== null) {
                handleUserCandidateChange(
                  selectedObjectIndex,
                  sessionIndex,
                  candidate,
                );
              }
            }}
          />
        </div>
      </div>

      {/* 2height: DB version create band, below the 3-panel screen */}
      <div className="mt-6">
        <CreateVersionPanel
          files={files}
          sourceFile={createSourceFile}
          onSourceFileChange={setCreateSourceFile}
          objects={createObjects}
          predictionMap={createPredictionMap}
          versions={versions}
          onCreated={refetchVersions}
          onViewVersion={setViewVersion}
        />
      </div>

      {/* Version-scoped contents viewer, opened by the eye icon in VersionSelect */}
      <BimAttributeTableModal
        version={viewVersion ?? undefined}
        open={viewVersion !== null}
        onOpenChange={(o) => {
          if (!o) setViewVersion(null);
        }}
      />
    </main>
  );
}
