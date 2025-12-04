import { memo, useMemo } from 'react';
import { ChatSession, RetrievalSettings, UploadedDocument } from '@/types';
import { BookOpen, Plus, MessageSquare, Trash2, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { Settings } from './Settings';
import { Button } from 'react-aria-components';
import { cn } from '@/lib/utils';
import { useChatSessionContext } from '@/contexts/ChatSessionContext';

interface SidebarProps {
  currentDocument?: UploadedDocument;
  onDocumentUploaded: (document: UploadedDocument) => void;
  onDocumentRemoved: () => void;
  onToggle: () => void;
  isExpanded: boolean;
  retrievalSettings?: RetrievalSettings;
  onRetrievalSettingsChange?: (settings: RetrievalSettings) => void;
}

function formatDate(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) {
    return 'Today';
  } else if (diffDays === 1) {
    return 'Yesterday';
  } else if (diffDays < 7) {
    return `${diffDays} days ago`;
  } else {
    return date.toLocaleDateString();
  }
}

const SessionItem = memo(function SessionItem({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session: ChatSession;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const formattedDate = useMemo(() => formatDate(session.updatedAt), [session.updatedAt]);

  return (
    <div
      className={cn(
        'group relative rounded-xl border transition-all cursor-pointer',
        isActive
          ? 'border-primary-200 dark:border-primary-500/30 bg-primary-50 dark:bg-primary-500/10 shadow-sm'
          : 'border-transparent hover:border-slate-200 dark:hover:border-dark-700 hover:bg-white dark:hover:bg-dark-850'
      )}
    >
      <button onClick={onSelect} className="w-full text-left p-3 pr-10">
        <div className="flex items-start gap-3">
          <MessageSquare
            className={cn(
              'h-5 w-5 mt-0.5 shrink-0',
              isActive
                ? 'text-primary-600 dark:text-primary-400'
                : 'text-slate-400 dark:text-slate-500'
            )}
          />
          <div className="min-w-0 flex-1">
            <p
              className={cn(
                'text-sm font-medium truncate',
                isActive
                  ? 'text-primary-900 dark:text-primary-100'
                  : 'text-slate-900 dark:text-white'
              )}
            >
              {session.title}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              {session.messages.length} messages · {formattedDate}
            </p>
          </div>
        </div>
      </button>
      <Button
        onPress={onDelete}
        className={cn(
          'absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg transition-all',
          'opacity-0 group-hover:opacity-100',
          'hover:bg-red-100 dark:hover:bg-red-500/20'
        )}
        aria-label="Delete chat"
      >
        <Trash2 className="h-4 w-4 text-red-500 dark:text-red-400" />
      </Button>
    </div>
  );
});

const CollapsedSidebar = memo(function CollapsedSidebar({
  currentDocument,
  onDocumentUploaded,
  onDocumentRemoved,
  onToggle,
  retrievalSettings,
  onRetrievalSettingsChange,
}: Omit<SidebarProps, 'isExpanded'>) {
  const { sessions, activeSessionId, session, createNewChat, switchSession } = useChatSessionContext();

  return (
    <div className="hidden md:flex h-full w-20 flex-col border-r border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-900">
      {/* Header with expand button */}
      <div className="flex items-center justify-center p-4 h-[85px] border-b border-slate-200 dark:border-dark-800 shrink-0">
        <Button
          onPress={onToggle}
          className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
          aria-label="Expand sidebar"
        >
          <PanelLeftOpen className="h-5 w-5 text-slate-600 dark:text-slate-400" />
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="p-3 border-b border-slate-200 dark:border-dark-800 shrink-0">
        <Button
          onPress={createNewChat}
          className="flex w-full items-center justify-center p-3 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 text-white hover:shadow-lg hover:shadow-primary-500/30 transition-all"
          aria-label="New chat"
        >
          <Plus className="h-5 w-5" />
        </Button>
      </div>

      {/* Recent chats as icons */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {sessions.slice(0, 10).map((s) => (
          <Button
            key={s.id}
            onPress={() => switchSession(s.id)}
            className={cn(
              'flex w-full items-center justify-center p-3 rounded-xl transition-all',
              s.id === activeSessionId
                ? 'bg-primary-50 dark:bg-primary-500/10 text-primary-600 dark:text-primary-400'
                : 'hover:bg-slate-100 dark:hover:bg-dark-800 text-slate-500 dark:text-slate-400'
            )}
            aria-label={s.title}
          >
            <MessageSquare className="h-5 w-5" />
          </Button>
        ))}
      </div>

      {/* Settings icon */}
      <div className="px-3 py-5 border-t border-slate-200 dark:border-dark-800 shrink-0 h-[85px]">
        <Settings
          sessionId={session.id}
          currentDocument={currentDocument}
          onDocumentUploaded={onDocumentUploaded}
          onDocumentRemoved={onDocumentRemoved}
          retrievalSettings={retrievalSettings}
          onRetrievalSettingsChange={onRetrievalSettingsChange}
          collapsed={true}
        />
      </div>
    </div>
  );
});

const ExpandedSidebar = memo(function ExpandedSidebar({
  currentDocument,
  onDocumentUploaded,
  onDocumentRemoved,
  onToggle,
  retrievalSettings,
  onRetrievalSettingsChange,
}: Omit<SidebarProps, 'isExpanded'>) {
  const { sessions, activeSessionId, session, createNewChat, switchSession, deleteSession } =
    useChatSessionContext();

  return (
    <div className="flex h-full w-80 flex-col border-r border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-900 shadow-lg dark:shadow-none">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-dark-800 bg-gradient-to-br from-white to-slate-50 dark:from-dark-850 dark:to-dark-900 px-4 py-4 shrink-0 flex items-center justify-between h-[85px]">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-500/30">
            <BookOpen className="h-6 w-6 text-white" />
          </div>
          <div className="overflow-hidden">
            <h1 className="text-lg font-bold text-slate-900 dark:text-white truncate">RAG System</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 truncate">Chat History</p>
          </div>
        </div>
        <Button
          onPress={onToggle}
          className="shrink-0 rounded-lg p-2 hover:bg-slate-200 dark:hover:bg-dark-800 transition-colors"
          aria-label="Collapse sidebar"
        >
          <PanelLeftClose className="h-5 w-5 text-slate-600 dark:text-slate-400" />
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="p-4 border-b border-slate-200 dark:border-dark-800 shrink-0">
        <Button
          onPress={createNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 px-4 py-3 text-sm font-semibold text-white hover:shadow-lg hover:shadow-primary-500/30 transition-all hover:scale-[1.02]"
        >
          <Plus className="h-5 w-5" />
          New Chat
        </Button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto scrollbar-thin bg-slate-50/50 dark:bg-dark-900 p-4">
        {sessions.length === 0 ? (
          <div className="text-center py-8">
            <MessageSquare className="h-12 w-12 text-slate-300 dark:text-dark-700 mx-auto mb-3" />
            <p className="text-sm text-slate-500 dark:text-slate-400">No chat history</p>
          </div>
        ) : (
          <div className="space-y-2">
            {sessions.map((s) => (
              <SessionItem
                key={s.id}
                session={s}
                isActive={s.id === activeSessionId}
                onSelect={() => switchSession(s.id)}
                onDelete={() => deleteSession(s.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer with Settings */}
      <div className="border-t border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 py-4 shrink-0 h-[85px]">
        <Settings
          sessionId={session.id}
          currentDocument={currentDocument}
          onDocumentUploaded={onDocumentUploaded}
          onDocumentRemoved={onDocumentRemoved}
          retrievalSettings={retrievalSettings}
          onRetrievalSettingsChange={onRetrievalSettingsChange}
        />
      </div>
    </div>
  );
});

export const Sidebar = memo(function Sidebar(props: SidebarProps) {
  const { isExpanded, ...rest } = props;

  if (!isExpanded) {
    return <CollapsedSidebar {...rest} />;
  }

  return <ExpandedSidebar {...rest} />;
});
