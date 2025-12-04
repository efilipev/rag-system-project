import { memo, useState, useCallback, useMemo, ReactNode } from 'react';
import { Button } from 'react-aria-components';
import { PanelLeftOpen } from 'lucide-react';
import { useChatSessionContext } from '@/contexts/ChatSessionContext';
import { Sidebar } from './Sidebar';
import { RetrievalSettings, UploadedDocument } from '@/types';
import { cn } from '@/lib/utils';
import { apiClient } from '@/lib/api';

interface LayoutProps {
  children: ReactNode;
  retrievalSettings: RetrievalSettings;
  onRetrievalSettingsChange: (settings: RetrievalSettings) => void;
}

export const Layout = memo(function Layout({
  children,
  retrievalSettings,
  onRetrievalSettingsChange,
}: LayoutProps) {
  const { session, sessions, activeSessionId } = useChatSessionContext();

  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [document, setDocument] = useState<UploadedDocument | undefined>();

  const handleDocumentUploaded = useCallback((doc: UploadedDocument) => {
    setDocument(doc);
    if (doc.collection) {
      onRetrievalSettingsChange({
        ...retrievalSettings,
        collection: doc.collection,
      });
    }
  }, [onRetrievalSettingsChange, retrievalSettings]);

  const handleDocumentRemoved = useCallback(async () => {
    if (document) {
      try {
        await apiClient.deleteDocument(document.id, session.id);
      } catch (error) {
        console.error('Failed to delete document:', error);
      }
    }
    setDocument(undefined);
  }, [document, session.id]);

  const handleToggleSidebar = useCallback(() => {
    setSidebarExpanded(prev => !prev);
  }, []);

  const handleCloseMobileOverlay = useCallback(() => {
    setSidebarExpanded(false);
  }, []);

  const handleOpenMobileSidebar = useCallback(() => {
    setSidebarExpanded(true);
  }, []);

  const currentSessionTitle = useMemo(() => {
    return sessions.find(s => s.id === activeSessionId)?.title || 'New Chat';
  }, [sessions, activeSessionId]);

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-dark-950">
      {/* Sidebar Overlay for Mobile when expanded */}
      {sidebarExpanded && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={handleCloseMobileOverlay}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'h-full transition-all duration-300 ease-in-out shrink-0',
          'fixed inset-y-0 left-0 z-50 md:relative md:z-auto',
          sidebarExpanded
            ? 'w-80 translate-x-0'
            : 'w-0 -translate-x-full md:w-20 md:translate-x-0'
        )}
      >
        <Sidebar
          currentDocument={document}
          onDocumentUploaded={handleDocumentUploaded}
          onDocumentRemoved={handleDocumentRemoved}
          onToggle={handleToggleSidebar}
          isExpanded={sidebarExpanded}
          retrievalSettings={retrievalSettings}
          onRetrievalSettingsChange={onRetrievalSettingsChange}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Mobile top bar */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-900 md:hidden shrink-0">
          <Button
            onPress={handleOpenMobileSidebar}
            className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
          >
            <PanelLeftOpen className="h-5 w-5 text-slate-600 dark:text-slate-400" />
          </Button>
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">
            {currentSessionTitle}
          </span>
        </div>

        {/* Content Area */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {children}
        </div>
      </div>
    </div>
  );
});
