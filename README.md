# RAG System - Microservices Backend

Retrieval-Augmented Generation (RAG) system built with microservices architecture, using Docker, LangChain, Qdrant, Chroma, and Poetry.

## Architecture Overview

This system consists of multiple microservices working together to provide intelligent document retrieval and response generation:

- **API Gateway** (Kong): Routes requests, handles authentication, rate limiting
- **Query Analysis Service**: Analyzes user queries using NLP and LangChain
- **Document Retrieval Service**: Retrieves relevant documents using Qdrant and Chroma vector databases
- **Document Ranking Service**: Ranks and reranks documents using cross-encoder models
- **LaTeX Parser Service**: Processes and parses mathematical LaTeX formulas
- **LLM Generation Service**: Generates responses using LLMs (OpenAI or local models)
- **Response Formatter Service**: Formats and structures the final response

### Infrastructure Components

- **PostgreSQL**: Primary database with pgvector extension
- **Qdrant**: High-performance vector database
- **Chroma**: Alternative vector database for hybrid search
- **Redis**: Caching and session management
- **RabbitMQ**: Message queue for async communication
- **MinIO**: S3-compatible object storage for documents
- **Prometheus + Grafana**: Monitoring and visualization