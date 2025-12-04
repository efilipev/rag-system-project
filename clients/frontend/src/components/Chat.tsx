import { memo, useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { Button } from 'react-aria-components';
import { Trash2, AlertCircle, Database } from 'lucide-react';
import { Message } from './Message';
import { ChatInput } from './ChatInput';
import { retrievalClient } from '@/lib/api';
import { DocumentSource, RetrievalSettings, DEFAULT_RETRIEVAL_SETTINGS, RETRIEVAL_METHOD_INFO } from '@/types';
import { useChatSessionContext } from '@/contexts/ChatSessionContext';

interface ChatProps {
  retrievalSettings?: RetrievalSettings;
}

const EmptyState = memo(function EmptyState({
  canChat,
  collection,
  retrievalMethod,
}: {
  canChat: boolean;
  collection?: string;
  retrievalMethod?: string;
}) {
  const methodName = retrievalMethod ? RETRIEVAL_METHOD_INFO[retrievalMethod as keyof typeof RETRIEVAL_METHOD_INFO]?.name || 'Hybrid Search' : 'Hybrid Search';

  if (canChat) {
    return (
      <div className="flex h-full items-center justify-center px-4">
        <div className="max-w-md text-center">
          <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-100 to-primary-50 dark:from-primary-500/20 dark:to-primary-500/5 shadow-lg shadow-primary-100/50 dark:shadow-none ring-1 ring-primary-200 dark:ring-primary-500/20">
            <Database className="h-10 w-10 text-primary-600 dark:text-primary-400" />
          </div>
          <h3 className="text-xl font-bold text-slate-900 dark:text-white">
            Connected to {collection}
          </h3>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
            Ask me anything about the {collection} knowledge base.
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
            Using {methodName} retrieval
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full items-center justify-center px-4">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-100 to-amber-50 dark:from-amber-500/20 dark:to-amber-500/5 shadow-lg shadow-amber-100/50 dark:shadow-none ring-1 ring-amber-200 dark:ring-amber-500/20">
          <AlertCircle className="h-10 w-10 text-amber-600 dark:text-amber-400" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 dark:text-white">
          Select a collection
        </h3>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Select a knowledge base collection from Settings to begin asking questions.
        </p>
      </div>
    </div>
  );
});

const ChatHeader = memo(function ChatHeader({
  messageCount,
  onClearChat,
}: {
  messageCount: number;
  onClearChat: () => void;
}) {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 px-6 py-4 shrink-0 shadow-sm dark:shadow-none h-[85px]">
      <div>
        <h2 className="text-lg font-bold text-slate-900 dark:text-white">Chat</h2>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {messageCount > 0
            ? `${messageCount} message${messageCount !== 1 ? 's' : ''}`
            : 'Start a conversation'}
        </p>
      </div>

      {messageCount > 0 && (
        <Button
          onPress={onClearChat}
          className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors border border-transparent hover:border-slate-200 dark:hover:border-dark-700"
        >
          <Trash2 className="h-4 w-4" />
          Clear Chat
        </Button>
      )}
    </div>
  );
});

export const Chat = memo(function Chat({ retrievalSettings }: ChatProps) {
  const { messages, addMessage, updateMessage, clearCurrentChat } = useChatSessionContext();

  const [isLoading, setIsLoading] = useState(false);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isUserScrollingRef = useRef(false);
  const lastMessageCountRef = useRef(0);

  const hasCollection = !!retrievalSettings?.collection;
  const canChat = hasCollection;

  // Smooth scroll to bottom
  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    if (isUserScrollingRef.current) return;

    requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior, block: 'end' });
    });
  }, []);

  // Scroll on new message or streaming update
  useEffect(() => {
    const messageCount = messages.length;
    const lastMessage = messages[messageCount - 1];

    // New message added - scroll immediately
    if (messageCount > lastMessageCountRef.current) {
      lastMessageCountRef.current = messageCount;
      isUserScrollingRef.current = false;
      scrollToBottom('instant');
      return;
    }

    // Streaming update - smooth scroll if not user scrolling
    if (lastMessage?.isStreaming) {
      scrollToBottom('smooth');
    }
  }, [messages, scrollToBottom]);

  // Track user scroll
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const { scrollHeight, scrollTop, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // User is considered scrolling if they're more than 100px from bottom
    isUserScrollingRef.current = distanceFromBottom > 100;
  }, []);

  const handleSendMessage = useCallback(async (content: string) => {
    if (!canChat) return;

    addMessage({
      role: 'user',
      content,
    });

    setIsLoading(true);

    const assistantMessage = addMessage({
      role: 'assistant',
      content: '',
      isStreaming: true,
    });

    try {
      const targetCollection = retrievalSettings?.collection || '';
      let fullResponse = '';
      let sources: DocumentSource[] = [];

      // Build request with retrieval method settings
      const retrievalMethod = retrievalSettings?.retrievalMethod || DEFAULT_RETRIEVAL_SETTINGS.retrievalMethod;

      for await (const chunk of retrievalClient.queryStream({
        query: content,
        collection: targetCollection,
        top_k: retrievalSettings?.topK || DEFAULT_RETRIEVAL_SETTINGS.topK,
        score_threshold: retrievalSettings?.scoreThreshold || DEFAULT_RETRIEVAL_SETTINGS.scoreThreshold,
        enable_query_analysis: retrievalSettings?.enableQueryAnalysis ?? DEFAULT_RETRIEVAL_SETTINGS.enableQueryAnalysis,
        enable_ranking: retrievalSettings?.enableRanking ?? DEFAULT_RETRIEVAL_SETTINGS.enableRanking,
        model: retrievalSettings?.model || DEFAULT_RETRIEVAL_SETTINGS.model,
        // New retrieval method parameter
        retrieval_method: retrievalMethod,
        // Hybrid search options
        ...(retrievalMethod === 'hybrid' && {
          hybrid_options: {
            dense_weight: retrievalSettings?.hybridOptions?.denseWeight || DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.denseWeight,
            sparse_weight: retrievalSettings?.hybridOptions?.sparseWeight || DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.sparseWeight,
            fusion_method: retrievalSettings?.hybridOptions?.fusionMethod || DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.fusionMethod,
          },
        }),
        // HyDE-ColBERT options
        use_hyde_colbert: retrievalMethod === 'hyde_colbert',
        ...(retrievalMethod === 'hyde_colbert' && {
          hyde_colbert_options: {
            n_hypotheticals: retrievalSettings?.hydeColbertOptions?.nHypotheticals || DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.nHypotheticals,
            domain: retrievalSettings?.hydeColbertOptions?.domain || DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.domain,
            fusion_strategy: retrievalSettings?.hydeColbertOptions?.fusionStrategy || DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.fusionStrategy,
            fusion_weight: retrievalSettings?.hydeColbertOptions?.fusionWeight || DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.fusionWeight,
            quality_threshold: retrievalSettings?.hydeColbertOptions?.qualityThreshold || DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.qualityThreshold,
          },
        }),
      })) {
        if (chunk.type === 'sources' && chunk.sources) {
          sources = chunk.sources.map((src, idx) => ({
            id: `${idx}`,
            title: src.title,
            content: src.content,
            score: src.score,
            metadata: src.metadata,
          }));
        } else if (chunk.type === 'token' && chunk.content) {
          fullResponse += chunk.content;
          updateMessage(assistantMessage.id, {
            content: fullResponse,
            isStreaming: true,
            sources,
          });
        } else if (chunk.type === 'done') {
          updateMessage(assistantMessage.id, {
            content: fullResponse || 'No response generated.',
            isStreaming: false,
            sources,
          });
        } else if (chunk.type === 'error') {
          updateMessage(assistantMessage.id, {
            content: `Error: ${chunk.error || 'Unknown error'}`,
            isStreaming: false,
            sources,
          });
        }
      }

      if (fullResponse) {
        updateMessage(assistantMessage.id, {
          content: fullResponse,
          isStreaming: false,
          sources,
        });
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      updateMessage(assistantMessage.id, {
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
        isStreaming: false,
      });
    } finally {
      setIsLoading(false);
    }
  }, [canChat, addMessage, updateMessage, retrievalSettings]);

  const handleClearChat = useCallback(() => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      clearCurrentChat();
      lastMessageCountRef.current = 0;
    }
  }, [clearCurrentChat]);

  const placeholder = useMemo(() => {
    return canChat
      ? `Ask about ${retrievalSettings?.collection}...`
      : 'Select a collection to start';
  }, [canChat, retrievalSettings?.collection]);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <ChatHeader messageCount={messages.length} onClearChat={handleClearChat} />

      {/* Messages container - takes all available space */}
      <div
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto scrollbar-thin bg-slate-50/50 dark:bg-dark-950"
      >
        {messages.length === 0 ? (
          <EmptyState
            canChat={canChat}
            collection={retrievalSettings?.collection}
            retrievalMethod={retrievalSettings?.retrievalMethod}
          />
        ) : (
          <div className="divide-y divide-slate-200 dark:divide-dark-800/50">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} className="h-0" />
          </div>
        )}
      </div>

      {/* Input - fixed at bottom */}
      <ChatInput
        onSend={handleSendMessage}
        disabled={!canChat}
        isLoading={isLoading}
        placeholder={placeholder}
      />
    </div>
  );
});
