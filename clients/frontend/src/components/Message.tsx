import { memo, useMemo } from 'react';
import { Message as MessageType } from '@/types';
import { User, Bot, FileText, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import { formatTimestamp } from '@/lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface MessageProps {
  message: MessageType;
}

const MessageComponent = memo(function MessageComponent({ message }: MessageProps) {
  const isUser = message.role === 'user';
  const hasSources = message.sources && message.sources.length > 0;

  const formattedTime = useMemo(
    () => formatTimestamp(message.timestamp),
    [message.timestamp]
  );

  const markdownComponents = useMemo(
    () => ({
      p: ({ children }: { children?: React.ReactNode }) => (
        <p className="mb-2 last:mb-0 text-slate-900 dark:text-slate-100">{children}</p>
      ),
      code: ({ inline, children, ...props }: any) =>
        inline ? (
          <code
            className="rounded bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 text-sm font-mono text-slate-900 dark:text-slate-100"
            {...props}
          >
            {children}
          </code>
        ) : (
          <pre className="rounded-lg bg-slate-900 dark:bg-slate-950 p-4 overflow-x-auto">
            <code className="text-sm font-mono text-slate-100" {...props}>
              {children}
            </code>
          </pre>
        ),
      a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 underline"
        >
          {children}
        </a>
      ),
    }),
    []
  );

  return (
    <div
      className={cn(
        'flex gap-4 px-6 py-6',
        isUser ? 'bg-white dark:bg-dark-900' : 'bg-slate-50/80 dark:bg-dark-900/50'
      )}
    >
      <div
        className={cn(
          'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl shadow-sm',
          isUser
            ? 'bg-gradient-to-br from-primary-500 to-primary-600 shadow-primary-500/30'
            : 'bg-gradient-to-br from-slate-700 to-slate-800 dark:from-dark-700 dark:to-dark-800 shadow-slate-500/20 dark:shadow-none'
        )}
      >
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-white" />
        )}
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <span className="text-sm font-bold text-slate-900 dark:text-white">
            {isUser ? 'You' : 'Assistant'}
          </span>
          <span className="text-xs text-slate-400 dark:text-slate-500">
            {formattedTime}
          </span>
        </div>

        {/* Text Content */}
        <div className="prose prose-slate dark:prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={markdownComponents}
          >
            {message.content}
          </ReactMarkdown>
          {message.isStreaming && (
            <span className="inline-block h-5 w-2 bg-slate-900 dark:bg-slate-100 cursor-blink ml-1" />
          )}
        </div>

        {/* Sources - always at bottom of message */}
        {hasSources && (
          <div className="mt-4 pt-4 border-t border-slate-200 dark:border-dark-800">
            <div className="flex items-center gap-2 text-sm font-bold text-slate-700 dark:text-slate-300 mb-2">
              <FileText className="h-4 w-4" />
              <span>Sources ({message.sources!.length})</span>
            </div>
            <div className="space-y-2">
              {message.sources!.map((source, idx) => {
                // Get destination (URL or page anchor) - check source.destination first, then metadata
                const destination = source.destination || source.metadata?.url;
                // Clamp score between 0 and 1 for display
                const displayScore = Math.min(1, Math.max(0, source.score));
                // Get page info (for PDFs) or chunk index (for other docs)
                const pageNum = source.metadata?.page;
                const totalPages = source.metadata?.total_pages;
                const chunkIndex = source.metadata?.chunk_index;
                // Determine if this is a PDF (has page info or filename ends with .pdf)
                const isPDF = pageNum !== undefined || source.title?.toLowerCase().endsWith('.pdf');

                return (
                  <div
                    key={source.id}
                    className="rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-3 text-sm shadow-sm dark:shadow-none hover:shadow-md dark:hover:border-dark-700 transition-all"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          {destination ? (
                            <a
                              href={destination}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 hover:underline truncate flex items-center gap-1"
                            >
                              {source.title || `Source ${idx + 1}`}
                              <ExternalLink className="h-3 w-3 flex-shrink-0" />
                            </a>
                          ) : (
                            <p className="font-semibold text-slate-900 dark:text-white truncate">
                              {source.title || `Source ${idx + 1}`}
                            </p>
                          )}
                          {/* Show page number for PDFs, chunk index for other documents */}
                          {isPDF && pageNum !== undefined ? (
                            <span className="text-xs text-slate-500 dark:text-slate-500 flex-shrink-0">
                              (page {pageNum}{totalPages ? ` of ${totalPages}` : ''})
                            </span>
                          ) : chunkIndex !== undefined ? (
                            <span className="text-xs text-slate-500 dark:text-slate-500 flex-shrink-0">
                              (chunk {chunkIndex})
                            </span>
                          ) : null}
                        </div>
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400 line-clamp-2">
                          {source.content}
                        </p>
                      </div>
                      <span className="flex-shrink-0 rounded-lg bg-primary-100 dark:bg-primary-500/20 px-2 py-1 text-xs font-bold text-primary-700 dark:text-primary-400 ring-1 ring-primary-200 dark:ring-primary-500/30">
                        {Math.round(displayScore * 100)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export { MessageComponent as Message };
