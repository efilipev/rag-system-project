import { useState, useCallback, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ChatSession, Message } from '@/types';

const STORAGE_KEY = 'rag-chat-history';
const MAX_SESSIONS = 50;

interface ChatHistoryState {
  sessions: ChatSession[];
  activeSessionId: string | null;
}

function createNewSession(): ChatSession {
  const now = new Date();
  return {
    id: uuidv4(),
    title: 'New Chat',
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

function generateTitle(messages: Message[]): string {
  const firstUserMessage = messages.find(m => m.role === 'user');
  if (!firstUserMessage) return 'New Chat';

  // Truncate to first 50 characters
  const content = firstUserMessage.content.trim();
  if (content.length <= 50) return content;
  return content.substring(0, 47) + '...';
}

function loadFromStorage(): ChatHistoryState {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      const sessions = parsed.sessions.map((s: any) => ({
        ...s,
        createdAt: new Date(s.createdAt),
        updatedAt: new Date(s.updatedAt),
        messages: s.messages.map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp),
        })),
      }));
      return {
        sessions,
        activeSessionId: parsed.activeSessionId,
      };
    }
  } catch (e) {
    console.error('Failed to load chat history:', e);
  }

  // Create initial session
  const initialSession = createNewSession();
  return {
    sessions: [initialSession],
    activeSessionId: initialSession.id,
  };
}

function saveToStorage(state: ChatHistoryState): void {
  try {
    // Limit stored sessions
    const sessionsToStore = state.sessions.slice(0, MAX_SESSIONS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      sessions: sessionsToStore,
      activeSessionId: state.activeSessionId,
    }));
  } catch (e) {
    console.warn('Failed to save chat history:', e);
    if (e instanceof DOMException && e.name === 'QuotaExceededError') {
      // Clear old sessions and try again
      const reducedSessions = state.sessions.slice(0, 10);
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sessions: reducedSessions,
        activeSessionId: state.activeSessionId,
      }));
    }
  }
}

export function useChatSession() {
  const [state, setState] = useState<ChatHistoryState>(loadFromStorage);

  // Persist to localStorage on changes
  useEffect(() => {
    saveToStorage(state);
  }, [state]);

  const activeSession = state.sessions.find(s => s.id === state.activeSessionId) || state.sessions[0];

  const createNewChat = useCallback(() => {
    const newSession = createNewSession();
    setState(prev => ({
      sessions: [newSession, ...prev.sessions],
      activeSessionId: newSession.id,
    }));
    return newSession;
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    setState(prev => ({
      ...prev,
      activeSessionId: sessionId,
    }));
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setState(prev => {
      const newSessions = prev.sessions.filter(s => s.id !== sessionId);

      // If we're deleting the active session, switch to another one
      let newActiveId = prev.activeSessionId;
      if (sessionId === prev.activeSessionId) {
        if (newSessions.length > 0) {
          newActiveId = newSessions[0].id;
        } else {
          // Create a new session if we deleted the last one
          const newSession = createNewSession();
          newSessions.push(newSession);
          newActiveId = newSession.id;
        }
      }

      return {
        sessions: newSessions,
        activeSessionId: newActiveId,
      };
    });
  }, []);

  const addMessage = useCallback((message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: uuidv4(),
      timestamp: new Date(),
    };

    setState(prev => {
      const sessions = prev.sessions.map(session => {
        if (session.id !== prev.activeSessionId) return session;

        const updatedMessages = [...session.messages, newMessage];
        const title = session.messages.length === 0 && message.role === 'user'
          ? generateTitle([newMessage])
          : session.title;

        return {
          ...session,
          messages: updatedMessages,
          title,
          updatedAt: new Date(),
        };
      });

      return { ...prev, sessions };
    });

    return newMessage;
  }, []);

  const updateMessage = useCallback((messageId: string, updates: Partial<Message>) => {
    setState(prev => {
      const sessions = prev.sessions.map(session => {
        if (session.id !== prev.activeSessionId) return session;

        return {
          ...session,
          messages: session.messages.map(msg =>
            msg.id === messageId ? { ...msg, ...updates } : msg
          ),
          updatedAt: new Date(),
        };
      });

      return { ...prev, sessions };
    });
  }, []);

  const clearCurrentChat = useCallback(() => {
    setState(prev => {
      const sessions = prev.sessions.map(session => {
        if (session.id !== prev.activeSessionId) return session;

        return {
          ...session,
          messages: [],
          title: 'New Chat',
          updatedAt: new Date(),
        };
      });

      return { ...prev, sessions };
    });
  }, []);

  const clearAllHistory = useCallback(() => {
    const newSession = createNewSession();
    setState({
      sessions: [newSession],
      activeSessionId: newSession.id,
    });
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return {
    // Current session
    session: activeSession,
    messages: activeSession?.messages || [],

    // All sessions for history
    sessions: state.sessions,
    activeSessionId: state.activeSessionId,

    // Actions
    addMessage,
    updateMessage,
    createNewChat,
    switchSession,
    deleteSession,
    clearCurrentChat,
    clearAllHistory,
  };
}
