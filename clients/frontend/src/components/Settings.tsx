import { memo, useState, useEffect, useCallback } from 'react';
import { Button, Dialog, DialogTrigger, Heading, Modal, ModalOverlay } from 'react-aria-components';
import { Settings as SettingsIcon, X, Sun, Moon, Database, Zap, RefreshCw, ChevronDown, ChevronUp, Layers, Search, Brain, Sparkles } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { DocumentUpload } from './DocumentUpload';
import { UploadedDocument, Collection, RetrievalSettings, RetrievalMethod, RETRIEVAL_METHOD_INFO, DEFAULT_RETRIEVAL_SETTINGS, HyDEDomain, FusionStrategy } from '@/types';
import { cn } from '@/lib/utils';
import { retrievalClient } from '@/lib/api';

interface SettingsProps {
  sessionId: string;
  currentDocument?: UploadedDocument;
  onDocumentUploaded: (document: UploadedDocument) => void;
  onDocumentRemoved: () => void;
  retrievalSettings?: RetrievalSettings;
  onRetrievalSettingsChange?: (settings: RetrievalSettings) => void;
  collapsed?: boolean;
}

const ThemeSection = memo(function ThemeSection() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
        Appearance
      </h3>
      <div className="flex items-center justify-between rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-4 shadow-sm dark:shadow-none hover:shadow-md transition-all">
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-yellow-100 to-yellow-50 dark:from-blue-500/20 dark:to-blue-500/5">
            {theme === 'light' ? (
              <Sun className="h-5 w-5 text-yellow-600" />
            ) : (
              <Moon className="h-5 w-5 text-blue-400" />
            )}
          </div>
          <div>
            <p className="text-sm font-bold text-slate-900 dark:text-white">Theme</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {theme === 'light' ? 'Light mode' : 'Dark mode'}
            </p>
          </div>
        </div>
        <Button
          onPress={toggleTheme}
          className={cn(
            'relative inline-flex h-7 w-12 items-center rounded-full transition-all shadow-inner',
            theme === 'dark' ? 'bg-gradient-to-r from-primary-500 to-primary-600' : 'bg-slate-300'
          )}
        >
          <span
            className={cn(
              'inline-block h-5 w-5 transform rounded-full bg-white shadow-md transition-transform',
              theme === 'dark' ? 'translate-x-6' : 'translate-x-1'
            )}
          />
        </Button>
      </div>
    </div>
  );
});

const DocumentSection = memo(function DocumentSection({
  sessionId,
  currentDocument,
  onDocumentUploaded,
  onDocumentRemoved,
}: {
  sessionId: string;
  currentDocument?: UploadedDocument;
  onDocumentUploaded: (document: UploadedDocument) => void;
  onDocumentRemoved: () => void;
}) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
        Document
      </h3>
      <p className="text-xs text-slate-500 dark:text-slate-400">
        Upload a PDF document to start asking questions
      </p>
      <DocumentUpload
        sessionId={sessionId}
        currentDocument={currentDocument}
        onDocumentUploaded={onDocumentUploaded}
        onDocumentRemoved={onDocumentRemoved}
      />
    </div>
  );
});

// Icon mapping for retrieval methods
const RETRIEVAL_METHOD_ICONS: Record<RetrievalMethod, React.ReactNode> = {
  hybrid: <Layers className="h-5 w-5" />,
  dense: <Search className="h-5 w-5" />,
  hyde_colbert: <Brain className="h-5 w-5" />,
  rag_fusion: <Sparkles className="h-5 w-5" />,
};

const RetrievalMethodSection = memo(function RetrievalMethodSection({
  collections,
  loadingCollections,
  settings,
  onRefresh,
  onSettingsChange,
}: {
  collections: Collection[];
  loadingCollections: boolean;
  settings?: RetrievalSettings;
  onRefresh: () => void;
  onSettingsChange: (settings: RetrievalSettings) => void;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const currentMethod = settings?.retrievalMethod || 'hybrid';
  const methodInfo = RETRIEVAL_METHOD_INFO[currentMethod];

  const handleMethodChange = useCallback((method: RetrievalMethod) => {
    if (settings) {
      onSettingsChange({
        ...settings,
        retrievalMethod: method,
        useHydeColbert: method === 'hyde_colbert',
      });
    }
  }, [settings, onSettingsChange]);

  const handleCollectionChange = useCallback((collectionName: string) => {
    if (settings) {
      onSettingsChange({
        ...settings,
        collection: collectionName,
      });
    }
  }, [settings, onSettingsChange]);

  // Hybrid options handlers
  const handleDenseWeightChange = useCallback((value: number) => {
    if (settings) {
      onSettingsChange({
        ...settings,
        hybridOptions: {
          ...DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!,
          ...settings.hybridOptions,
          denseWeight: value,
          sparseWeight: 1 - value,
        },
      });
    }
  }, [settings, onSettingsChange]);

  // HyDE-ColBERT options handlers
  const handleHydeOptionChange = useCallback((key: string, value: number | string) => {
    if (settings) {
      onSettingsChange({
        ...settings,
        hydeColbertOptions: {
          ...DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!,
          ...settings.hydeColbertOptions,
          [key]: value,
        },
      });
    }
  }, [settings, onSettingsChange]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
          Retrieval Settings
        </h3>
        <Button
          onPress={onRefresh}
          className="p-1.5 rounded-lg hover:bg-slate-200 dark:hover:bg-dark-800 transition-colors"
          aria-label="Refresh collections"
        >
          <RefreshCw className={cn('h-4 w-4 text-slate-500', loadingCollections && 'animate-spin')} />
        </Button>
      </div>

      {/* Collection Selector */}
      <div className="rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-4 shadow-sm dark:shadow-none">
        <div className="flex items-center space-x-3 mb-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-purple-100 to-purple-50 dark:from-purple-500/20 dark:to-purple-500/5">
            <Database className="h-5 w-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-bold text-slate-900 dark:text-white">Collection</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {collections.length > 0 ? `${collections.length} available` : 'No collections found'}
            </p>
          </div>
        </div>

        {loadingCollections ? (
          <div className="flex items-center justify-center py-4">
            <RefreshCw className="h-5 w-5 text-slate-400 animate-spin" />
            <span className="ml-2 text-sm text-slate-500">Loading collections...</span>
          </div>
        ) : collections.length > 0 ? (
          <select
            value={settings?.collection || ''}
            onChange={(e) => handleCollectionChange(e.target.value)}
            className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-dark-700 bg-white dark:bg-dark-800 text-slate-900 dark:text-white text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select a collection...</option>
            {collections.map((col) => (
              <option key={col.name} value={col.name}>
                {col.name} ({col.documentCount} docs)
              </option>
            ))}
          </select>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-400 py-2">
            No collections available. Make sure the retrieval service is running.
          </p>
        )}
      </div>

      {/* Retrieval Method Selector */}
      <div className="rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-4 shadow-sm dark:shadow-none">
        <div className="flex items-center space-x-3 mb-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-100 to-emerald-50 dark:from-emerald-500/20 dark:to-emerald-500/5">
            <Zap className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-bold text-slate-900 dark:text-white">Retrieval Method</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {methodInfo.description}
            </p>
          </div>
        </div>

        {/* Method Cards */}
        <div className="grid grid-cols-2 gap-2">
          {(Object.keys(RETRIEVAL_METHOD_INFO) as RetrievalMethod[]).map((method) => {
            const info = RETRIEVAL_METHOD_INFO[method];
            const isSelected = currentMethod === method;
            return (
              <button
                key={method}
                onClick={() => handleMethodChange(method)}
                className={cn(
                  'flex flex-col items-start p-3 rounded-lg border-2 transition-all text-left',
                  isSelected
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-500/10'
                    : 'border-slate-200 dark:border-dark-700 hover:border-slate-300 dark:hover:border-dark-600'
                )}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <span className={cn(
                    isSelected ? 'text-primary-600 dark:text-primary-400' : 'text-slate-500 dark:text-slate-400'
                  )}>
                    {RETRIEVAL_METHOD_ICONS[method]}
                  </span>
                  <span className={cn(
                    'text-sm font-semibold',
                    isSelected ? 'text-primary-700 dark:text-primary-300' : 'text-slate-700 dark:text-slate-300'
                  )}>
                    {info.name}
                  </span>
                  {info.badge && (
                    <span className={cn(
                      'text-[10px] px-1.5 py-0.5 rounded-full font-medium',
                      info.badge === 'Recommended'
                        ? 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-400'
                        : 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400'
                    )}>
                      {info.badge}
                    </span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center justify-between w-full px-4 py-2 rounded-lg text-sm text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
      >
        <span className="font-medium">Advanced Settings</span>
        {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {/* Advanced Settings Panel */}
      {showAdvanced && (
        <div className="rounded-xl border border-slate-200 dark:border-dark-800 bg-slate-50 dark:bg-dark-900 p-4 space-y-4">
          {/* Hybrid Search Options */}
          {currentMethod === 'hybrid' && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase">Hybrid Search Settings</p>
              <div>
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                  <span>Dense: {Math.round((settings?.hybridOptions?.denseWeight || 0.3) * 100)}%</span>
                  <span>Sparse (BM25): {Math.round((settings?.hybridOptions?.sparseWeight || 0.7) * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={(settings?.hybridOptions?.denseWeight || 0.3) * 100}
                  onChange={(e) => handleDenseWeightChange(parseInt(e.target.value) / 100)}
                  className="w-full h-2 bg-slate-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                  Benchmark optimal: 30% dense + 70% sparse
                </p>
              </div>
            </div>
          )}

          {/* HyDE-ColBERT Options */}
          {currentMethod === 'hyde_colbert' && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase">HyDE-ColBERT Settings</p>

              {/* Domain */}
              <div>
                <label className="text-xs text-slate-500 dark:text-slate-400 mb-1 block">Domain</label>
                <select
                  value={settings?.hydeColbertOptions?.domain || 'general'}
                  onChange={(e) => handleHydeOptionChange('domain', e.target.value as HyDEDomain)}
                  className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-dark-700 bg-white dark:bg-dark-800 text-slate-900 dark:text-white text-sm"
                >
                  <option value="general">General</option>
                  <option value="biomedical">Biomedical</option>
                  <option value="scientific">Scientific</option>
                  <option value="financial">Financial</option>
                  <option value="argumentative">Argumentative</option>
                </select>
              </div>

              {/* Fusion Strategy */}
              <div>
                <label className="text-xs text-slate-500 dark:text-slate-400 mb-1 block">Fusion Strategy</label>
                <select
                  value={settings?.hydeColbertOptions?.fusionStrategy || 'weighted_average'}
                  onChange={(e) => handleHydeOptionChange('fusionStrategy', e.target.value as FusionStrategy)}
                  className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-dark-700 bg-white dark:bg-dark-800 text-slate-900 dark:text-white text-sm"
                >
                  <option value="weighted_average">Weighted Average (Best: 59.39% NDCG)</option>
                  <option value="average_all">Average All</option>
                  <option value="average_hyde_only">Average HyDE Only</option>
                  <option value="max">Max Score</option>
                  <option value="rrf">Reciprocal Rank Fusion</option>
                </select>
              </div>

              {/* Hypotheticals Count */}
              <div>
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                  <span>Hypothetical Documents</span>
                  <span>{settings?.hydeColbertOptions?.nHypotheticals || 3}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="6"
                  value={settings?.hydeColbertOptions?.nHypotheticals || 3}
                  onChange={(e) => handleHydeOptionChange('nHypotheticals', parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                  Optimal: 3 (best diversity/quality tradeoff)
                </p>
              </div>

              {/* Fusion Weight */}
              <div>
                <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                  <span>HyDE Weight</span>
                  <span>{Math.round((settings?.hydeColbertOptions?.fusionWeight || 0.2) * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={(settings?.hydeColbertOptions?.fusionWeight || 0.2) * 100}
                  onChange={(e) => handleHydeOptionChange('fusionWeight', parseInt(e.target.value) / 100)}
                  className="w-full h-2 bg-slate-200 dark:bg-dark-700 rounded-lg appearance-none cursor-pointer accent-primary-500"
                />
                <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                  Optimal: 20% HyDE + 80% query
                </p>
              </div>
            </div>
          )}

          {/* RAG Fusion info */}
          {currentMethod === 'rag_fusion' && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase">RAG Fusion</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Automatically generates query variations (paraphrases, decomposition, step-back) and combines results using Reciprocal Rank Fusion (k=60).
              </p>
            </div>
          )}

          {/* Dense Vector info */}
          {currentMethod === 'dense' && (
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase">Dense Vector Search</p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Pure semantic search using BAAI/bge-base-en-v1.5 embeddings. Fast and effective for semantic similarity queries.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

const AboutSection = memo(function AboutSection() {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wide">
        About
      </h3>
      <div className="rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-4 shadow-sm dark:shadow-none">
        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
          This RAG (Retrieval-Augmented Generation) system allows you to upload PDF documents and ask
          questions about their content. The system analyzes your queries, retrieves relevant
          information, and generates accurate answers based on the document.
        </p>
      </div>
    </div>
  );
});

export const Settings = memo(function Settings({
  sessionId,
  currentDocument,
  onDocumentUploaded,
  onDocumentRemoved,
  retrievalSettings,
  onRetrievalSettingsChange,
  collapsed = false,
}: SettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [loadingCollections, setLoadingCollections] = useState(false);

  const fetchCollections = useCallback(async () => {
    setLoadingCollections(true);
    try {
      const cols = await retrievalClient.getCollections();
      setCollections(cols);
    } catch (error) {
      console.error('Failed to fetch collections:', error);
    } finally {
      setLoadingCollections(false);
    }
  }, []);

  // Fetch collections when dialog opens
  useEffect(() => {
    if (isOpen) {
      fetchCollections();
    }
  }, [isOpen, fetchCollections]);

  const handleRetrievalSettingsChange = useCallback(
    (settings: RetrievalSettings) => {
      if (onRetrievalSettingsChange) {
        onRetrievalSettingsChange(settings);
      }
    },
    [onRetrievalSettingsChange]
  );

  const handleOpen = useCallback(() => setIsOpen(true), []);

  return (
    <DialogTrigger>
      <Button
        onPress={handleOpen}
        className={cn(
          'flex items-center rounded-xl text-sm font-semibold transition-all',
          'hover:bg-slate-100 dark:hover:bg-dark-800 hover:shadow-sm',
          'text-slate-700 dark:text-slate-300 border border-transparent hover:border-slate-200 dark:hover:border-dark-700',
          collapsed ? 'w-full justify-center p-3' : 'w-full space-x-2 px-4 py-2.5'
        )}
        aria-label="Settings"
      >
        <SettingsIcon className="h-5 w-5" />
        {!collapsed && <span>Settings</span>}
      </Button>

      <ModalOverlay
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        className={cn(
          'fixed inset-0 z-50 bg-black/60 backdrop-blur-md',
          'data-[entering]:animate-in data-[entering]:fade-in',
          'data-[exiting]:animate-out data-[exiting]:fade-out'
        )}
      >
        <Modal
          className={cn(
            'fixed left-[50%] top-[50%] z-50 w-full max-w-2xl',
            'translate-x-[-50%] translate-y-[-50%]',
            'rounded-2xl border shadow-2xl',
            'bg-white dark:bg-dark-850',
            'border-slate-200 dark:border-dark-800',
            'data-[entering]:animate-in data-[entering]:fade-in data-[entering]:zoom-in-95',
            'data-[exiting]:animate-out data-[exiting]:fade-out data-[exiting]:zoom-out-95'
          )}
        >
          <Dialog className="outline-none">
            {({ close }) => (
              <div className="flex flex-col max-h-[85vh]">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-slate-200 dark:border-dark-800 px-6 py-5 bg-gradient-to-r from-white to-slate-50 dark:from-dark-850 dark:to-dark-900 shrink-0">
                  <Heading slot="title" className="text-xl font-bold text-slate-900 dark:text-white">
                    Settings
                  </Heading>
                  <Button
                    onPress={close}
                    className="rounded-lg p-1.5 hover:bg-slate-200 dark:hover:bg-dark-800 transition-colors"
                  >
                    <X className="h-5 w-5 text-slate-500 dark:text-slate-400" />
                  </Button>
                </div>

                {/* Content */}
                <div className="overflow-y-auto flex-1 p-6 space-y-6 bg-slate-50 dark:bg-dark-900">
                  <ThemeSection />

                  <DocumentSection
                    sessionId={sessionId}
                    currentDocument={currentDocument}
                    onDocumentUploaded={onDocumentUploaded}
                    onDocumentRemoved={onDocumentRemoved}
                  />

                  <RetrievalMethodSection
                    collections={collections}
                    loadingCollections={loadingCollections}
                    settings={retrievalSettings}
                    onRefresh={fetchCollections}
                    onSettingsChange={handleRetrievalSettingsChange}
                  />

                  <AboutSection />
                </div>

                {/* Footer */}
                <div className="border-t border-slate-200 dark:border-dark-800 px-6 py-4 bg-white dark:bg-dark-850 shrink-0 h-[85px]">
                  <div className="flex justify-end">
                    <Button
                      onPress={close}
                      className="rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 px-5 py-2.5 text-sm font-semibold text-white hover:shadow-lg hover:shadow-primary-500/30 transition-all hover:scale-105"
                    >
                      Close
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </Dialog>
        </Modal>
      </ModalOverlay>
    </DialogTrigger>
  );
});
