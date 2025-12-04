import { memo, useState, useRef, useEffect, useCallback } from 'react';
import { Button } from 'react-aria-components';
import { Send, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export const ChatInput = memo(function ChatInput({
  onSend,
  disabled = false,
  isLoading = false,
  placeholder = 'Ask a question about your document...',
}: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const resetTextarea = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, []);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();

      if (!message.trim() || disabled || isLoading) return;

      onSend(message.trim());
      setMessage('');
      resetTextarea();
    },
    [message, disabled, isLoading, onSend, resetTextarea]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [message]);

  const isDisabled = disabled || isLoading;
  const canSubmit = message.trim() && !isDisabled;

  return (
    <form onSubmit={handleSubmit} className="border-t border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 py-4 shadow-sm dark:shadow-none h-[85px]">
      <div className="mx-auto max-w-3xl">
        <div className="flex gap-3 items-start">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isDisabled}
              rows={1}
              className={cn(
                'w-full resize-none rounded-xl border border-slate-200 dark:border-dark-700 bg-white dark:bg-dark-900 px-4 py-3',
                'text-sm text-slate-900 dark:text-white placeholder:text-slate-400 dark:placeholder:text-slate-500',
                'focus:border-primary-500 dark:focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20 dark:focus:ring-primary-500/30',
                'disabled:cursor-not-allowed disabled:bg-slate-50 dark:disabled:bg-dark-850 disabled:text-slate-500',
                'transition-all shadow-sm dark:shadow-none'
              )}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
          </div>

          <Button
            type="submit"
            isDisabled={!canSubmit}
            className={cn(
              'flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl',
              'bg-gradient-to-br from-primary-500 to-primary-600 text-white transition-all shadow-lg shadow-primary-500/30',
              'hover:shadow-xl hover:shadow-primary-500/40 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
              'disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 dark:disabled:from-dark-700 dark:disabled:to-dark-700 disabled:shadow-none disabled:text-slate-500 disabled:scale-100'
            )}
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Send className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>
    </form>
  );
});
