import { QueryRequest, QueryResponse, StreamChunk, UploadedDocument, Collection, RetrievalRequest, RetrievalResponse, RetrievalMethod, DEFAULT_RETRIEVAL_SETTINGS } from '@/types';

// API URL - Python API Gateway (port 8000) with orchestrator handles RAG queries
// Features: parallel query analysis + retrieval, document ranking, LaTeX detection
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

class APIClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Query with streaming response
   */
  async *queryStream(request: QueryRequest): AsyncGenerator<StreamChunk> {
    try {
      const response = await fetch(`${this.baseUrl}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              yield { type: 'done' };
              return;
            }

            try {
              const parsed = JSON.parse(data) as StreamChunk;
              yield parsed;
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error);
      yield {
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Query without streaming (fallback)
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Upload document
   * Routes through api-gateway which proxies to document-retrieval
   */
  async uploadDocument(file: File, sessionId: string, collectionName?: string): Promise<UploadedDocument> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    if (collectionName) {
      formData.append('collection_name', collectionName);
    }

    const response = await fetch(`${this.baseUrl}/documents/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(errorData.detail || `Upload failed! status: ${response.status}`);
    }

    const data = await response.json();

    // Convert file to base64 for PDF preview
    const base64 = await this.fileToBase64(file);

    return {
      id: data.id || crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: file.type,
      uploadedAt: new Date(),
      base64,
      collection: data.collection,
      chunksCreated: data.chunks_created,
    };
  }

  /**
   * Delete document
   */
  async deleteDocument(documentId: string, sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/documents/${documentId}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ sessionId }),
    });

    if (!response.ok) {
      throw new Error(`Delete failed! status: ${response.status}`);
    }
  }

  /**
   * Get available collections
   */
  async getCollections(): Promise<Collection[]> {
    try {
      const response = await fetch(`${this.baseUrl}/collections`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data.collections || [];
    } catch (error) {
      console.error('Failed to fetch collections:', error);
      return [];
    }
  }

  /**
   * Retrieve documents from a collection
   */
  async retrieve(request: RetrievalRequest): Promise<RetrievalResponse> {
    const response = await fetch(`${this.baseUrl}/retrieve`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Full RAG query with streaming response
   * Routes through api-gateway which proxies to document-retrieval
   * Supports multiple retrieval methods: hybrid (default), dense, hyde_colbert, rag_fusion
   */
  async *ragQueryStream(request: {
    query: string;
    collection: string;
    top_k?: number;
    score_threshold?: number;
    // New retrieval method parameter
    retrieval_method?: RetrievalMethod;
    // Hybrid search options (for 'hybrid' method)
    hybrid_options?: {
      dense_weight: number;
      sparse_weight: number;
      fusion_method: 'weighted' | 'rrf';
    };
    // Legacy fields (kept for backward compatibility)
    use_hyde_colbert?: boolean;
    hyde_colbert_options?: {
      n_hypotheticals: number;
      domain: string;
      fusion_strategy: string;
      fusion_weight: number;
      quality_threshold?: number;
    };
    model?: string;
    enable_query_analysis?: boolean;
    enable_ranking?: boolean;
  }): AsyncGenerator<{
    type: 'token' | 'sources' | 'done' | 'error';
    content?: string;
    sources?: Array<{
      title: string;
      content: string;
      score: number;
      metadata: Record<string, any>;
    }>;
    error?: string;
  }> {
    try {
      // Determine retrieval method - default to 'hybrid' (best benchmark)
      const retrievalMethod = request.retrieval_method || 'hybrid';

      // Build request body with retrieval method parameters
      const requestBody: Record<string, unknown> = {
        query: request.query,
        collection: request.collection,
        top_k: request.top_k || DEFAULT_RETRIEVAL_SETTINGS.topK,
        score_threshold: request.score_threshold || DEFAULT_RETRIEVAL_SETTINGS.scoreThreshold,
        model: request.model || DEFAULT_RETRIEVAL_SETTINGS.model,
        enable_query_analysis: request.enable_query_analysis ?? DEFAULT_RETRIEVAL_SETTINGS.enableQueryAnalysis,
        enable_ranking: request.enable_ranking ?? DEFAULT_RETRIEVAL_SETTINGS.enableRanking,
        // New retrieval method parameter
        retrieval_method: retrievalMethod,
      };

      // Add method-specific options
      if (retrievalMethod === 'hybrid') {
        requestBody.hybrid_options = request.hybrid_options || {
          dense_weight: DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.denseWeight,
          sparse_weight: DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.sparseWeight,
          fusion_method: DEFAULT_RETRIEVAL_SETTINGS.hybridOptions!.fusionMethod,
        };
      } else if (retrievalMethod === 'hyde_colbert') {
        requestBody.use_hyde_colbert = true;
        requestBody.hyde_colbert_options = request.hyde_colbert_options || {
          n_hypotheticals: DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.nHypotheticals,
          domain: DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.domain,
          fusion_strategy: DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.fusionStrategy,
          fusion_weight: DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.fusionWeight,
          quality_threshold: DEFAULT_RETRIEVAL_SETTINGS.hydeColbertOptions!.qualityThreshold,
        };
      } else {
        // For 'dense' and 'rag_fusion', just pass the method
        requestBody.use_hyde_colbert = false;
      }

      const response = await fetch(`${this.baseUrl}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              yield { type: 'done' };
              return;
            }

            try {
              const parsed = JSON.parse(data);
              yield parsed;
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Query stream error:', error);
      yield {
        type: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Helper: Convert file to base64
   */
  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // Remove data:application/pdf;base64, prefix
      };
      reader.onerror = (error) => reject(error);
    });
  }
}

// Single client instance - all API calls go through api-gateway
export const apiClient = new APIClient();

// For backwards compatibility with existing code that uses retrievalClient
// Expose ragQueryStream as queryStream since that's what Chat.tsx expects
export const retrievalClient = {
  queryStream: apiClient.ragQueryStream.bind(apiClient),
  getCollections: apiClient.getCollections.bind(apiClient),
  retrieve: apiClient.retrieve.bind(apiClient),
  healthCheck: apiClient.healthCheck.bind(apiClient),
};
