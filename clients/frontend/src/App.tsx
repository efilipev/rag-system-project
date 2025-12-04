import { useState } from 'react';
import { Layout } from '@/components/Layout';
import { Chat } from '@/components/Chat';
import { RetrievalSettings, DEFAULT_RETRIEVAL_SETTINGS } from '@/types';

export default function App() {
  // Initialize with default settings - hybrid search is the best based on benchmarks
  const [retrievalSettings, setRetrievalSettings] = useState<RetrievalSettings>({
    collection: 'wikipedia',
    ...DEFAULT_RETRIEVAL_SETTINGS,
  });

  return (
    <Layout
      retrievalSettings={retrievalSettings}
      onRetrievalSettingsChange={setRetrievalSettings}
    >
      <Chat retrievalSettings={retrievalSettings} />
    </Layout>
  );
}
