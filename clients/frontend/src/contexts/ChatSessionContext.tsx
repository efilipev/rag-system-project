import { createContext, useContext, ReactNode } from 'react';
import { useChatSession } from '@/hooks/useChatSession';

type ChatSessionContextType = ReturnType<typeof useChatSession>;

const ChatSessionContext = createContext<ChatSessionContextType | null>(null);

export function ChatSessionProvider({ children }: { children: ReactNode }) {
  const chatSession = useChatSession();

  return (
    <ChatSessionContext.Provider value={chatSession}>
      {children}
    </ChatSessionContext.Provider>
  );
}

export function useChatSessionContext() {
  const context = useContext(ChatSessionContext);
  if (!context) {
    throw new Error('useChatSessionContext must be used within ChatSessionProvider');
  }
  return context;
}
