export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: DocumentSource[];
  isStreaming?: boolean;
}

export interface DocumentSource {
  id: string;
  title: string;
  content: string;
  score: number;
  // Unified destination for source navigation:
  // - Wikipedia: full URL (e.g., "https://en.wikipedia.org/wiki/Mozart")
  // - PDF: page anchor (e.g., "#page=5") or could be local file reference
  // Falls back to metadata.url if not set
  destination?: string;
  metadata?: Record<string, any>;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
  base64?: string;
  collection?: string;
  chunksCreated?: number;
}

export interface QueryRequest {
  query: string;
  sessionId: string;
  userId?: string;
  context?: Record<string, any>;
}

export interface QueryResponse {
  success: boolean;
  response: string;
  sources?: DocumentSource[];
  processingTime?: number;
  error?: string;
}

export interface StreamChunk {
  type: 'token' | 'sources' | 'done' | 'error';
  content?: string;
  sources?: DocumentSource[];
  error?: string;
}

export interface Collection {
  name: string;
  documentCount: number;
  description?: string;
}

// Retrieval method types based on benchmark results
export type RetrievalMethod = 'hybrid' | 'dense' | 'hyde_colbert' | 'rag_fusion';

export type HyDEDomain = 'general' | 'biomedical' | 'scientific' | 'financial' | 'argumentative';

export type FusionStrategy = 'weighted_average' | 'average_all' | 'average_hyde_only' | 'max' | 'rrf';

export interface HyDEColBERTOptions {
  nHypotheticals: number;  // 1-6, default: 3 (optimal)
  domain: HyDEDomain;  // default: 'general'
  fusionStrategy: FusionStrategy;  // default: 'weighted_average' (59.39% NDCG@10)
  fusionWeight: number;  // 0.0-1.0, default: 0.2 (20% HyDE, 80% query - optimal)
  qualityThreshold: number;  // 0.0-1.0, default: 0.7
}

export interface HybridSearchOptions {
  denseWeight: number;  // 0.0-1.0, default: 0.3 (benchmark: 30% dense)
  sparseWeight: number;  // 0.0-1.0, default: 0.7 (benchmark: 70% sparse/BM25)
  fusionMethod: 'weighted' | 'rrf';  // default: 'weighted'
}

export interface RetrievalSettings {
  collection: string;
  // New retrieval method selector - 'hybrid' is default (best benchmark)
  retrievalMethod: RetrievalMethod;
  // Method-specific options
  hybridOptions?: HybridSearchOptions;
  hydeColbertOptions?: HyDEColBERTOptions;
  // Legacy field (kept for backward compatibility, maps to hyde_colbert method)
  useHydeColbert: boolean;
  topK: number;
  scoreThreshold: number;
  enableQueryAnalysis?: boolean;  // Enable query analysis with LaTeX detection (default: true)
  enableRanking?: boolean;  // Enable document ranking for better relevance (default: true)
  model?: string;  // LLM model to use (default: llama3.2:1b)
}

// Default retrieval settings based on benchmarks
export const DEFAULT_RETRIEVAL_SETTINGS: Omit<RetrievalSettings, 'collection'> = {
  retrievalMethod: 'hybrid',  // Best benchmark: hybrid achieves best NDCG
  hybridOptions: {
    denseWeight: 0.3,  // 30% dense (vector similarity)
    sparseWeight: 0.7,  // 70% sparse (BM25) - best NDCG per benchmarks
    fusionMethod: 'weighted',
  },
  hydeColbertOptions: {
    nHypotheticals: 3,  // Optimal diversity/quality tradeoff
    domain: 'general',
    fusionStrategy: 'weighted_average',  // 59.39% NDCG@10 on SciFact
    fusionWeight: 0.2,  // 20% HyDE, 80% query - optimal
    qualityThreshold: 0.7,
  },
  useHydeColbert: false,
  topK: 10,
  scoreThreshold: 0.3,
  enableQueryAnalysis: true,
  enableRanking: true,
  model: 'llama3.2:1b',
};

// Human-readable descriptions for retrieval methods
export const RETRIEVAL_METHOD_INFO: Record<RetrievalMethod, { name: string; description: string; badge?: string }> = {
  hybrid: {
    name: 'Hybrid Search',
    description: 'Combines dense vectors (30%) + BM25 sparse search (70%). Best overall accuracy.',
    badge: 'Recommended',
  },
  dense: {
    name: 'Dense Vector',
    description: 'Pure vector similarity search using embeddings. Fast and good for semantic queries.',
  },
  hyde_colbert: {
    name: 'HyDE-ColBERT',
    description: 'Generates hypothetical documents + token-level matching. 59.39% NDCG@10 on SciFact.',
    badge: 'Advanced',
  },
  rag_fusion: {
    name: 'RAG Fusion',
    description: 'Multi-query retrieval with paraphrases and decomposition. Good for complex queries.',
  },
};

export interface RetrievalRequest {
  query: string;
  top_k: number;
  use_qdrant: boolean;
  use_chroma: boolean;
  use_hyde_colbert: boolean;
  collection: string;
  score_threshold: number;
  // New retrieval method parameters
  retrieval_method?: RetrievalMethod;
  hybrid_options?: {
    dense_weight: number;
    sparse_weight: number;
    fusion_method: 'weighted' | 'rrf';
  };
  hyde_colbert_options?: {
    n_hypotheticals: number;
    domain: string;
    fusion_strategy: string;
    fusion_weight: number;
    quality_threshold?: number;
  };
}

export interface RetrievalResponse {
  success: boolean;
  documents: Array<{
    content: string;
    score: number;
    metadata: Record<string, any>;
    source: string;
  }>;
  total_found: number;
  processing_time_ms: number;
  error?: string;
}
