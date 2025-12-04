"""
RAG Pipeline Orchestrator - Coordinates all microservices
"""
import asyncio
import logging
import uuid
import re
from typing import Dict, Any, List, Optional, Tuple
from shared.clients.service_clients import (
    QueryAnalysisClient,
    DocumentRetrievalClient,
    DocumentRankingClient,
    LatexParserClient,
    LLMGenerationClient,
    ResponseFormatterClient
)

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Orchestrator for RAG pipeline
    Coordinates query analysis, retrieval, ranking, parsing, generation, and formatting
    """

    def __init__(
        self,
        query_analysis_url: str,
        document_retrieval_url: str,
        document_ranking_url: str,
        latex_parser_url: str,
        llm_generation_url: str,
        response_formatter_url: str
    ):
        """
        Initialize orchestrator with service URLs

        Args:
            query_analysis_url: Query Analysis Service URL
            document_retrieval_url: Document Retrieval Service URL
            document_ranking_url: Document Ranking Service URL
            latex_parser_url: LaTeX Parser Service URL
            llm_generation_url: LLM Generation Service URL
            response_formatter_url: Response Formatter Service URL
        """
        # Initialize clients
        self.query_analysis_client = QueryAnalysisClient(base_url=query_analysis_url)
        self.document_retrieval_client = DocumentRetrievalClient(base_url=document_retrieval_url)
        self.document_ranking_client = DocumentRankingClient(base_url=document_ranking_url)
        self.latex_parser_client = LatexParserClient(base_url=latex_parser_url)
        self.llm_generation_client = LLMGenerationClient(base_url=llm_generation_url)
        self.response_formatter_client = ResponseFormatterClient(base_url=response_formatter_url)

        logger.info("RAG Orchestrator initialized with all service clients")

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracing"""
        return str(uuid.uuid4())

    def _detect_latex(self, text: str) -> bool:
        """
        Detect if text contains LaTeX formulas

        Args:
            text: Text to check

        Returns:
            True if LaTeX detected
        """
        # Check for common LaTeX patterns
        latex_patterns = [
            r'\$.*?\$',  # Inline math: $...$
            r'\\\[.*?\\\]',  # Display math: \[...\]
            r'\\begin\{equation\}',  # Equation environment
            r'\\frac\{',  # Fraction
            r'\\sum',  # Sum
            r'\\int',  # Integral
            r'\\alpha|\\beta|\\gamma',  # Greek letters
        ]

        for pattern in latex_patterns:
            if re.search(pattern, text):
                return True

        return False

    async def _parallel_query_analysis_and_retrieval(
        self,
        query: str,
        collection: str,
        retrieval_top_k: int,
        score_threshold: float,
        use_hyde_colbert: bool,
        hyde_colbert_options: Optional[Dict[str, Any]],
        enable_query_analysis: bool,
        correlation_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Run Query Analysis and Document Retrieval in parallel for better performance.

        Returns:
            Tuple of (analysis_result, retrieval_result)
        """
        async def do_analysis():
            if not enable_query_analysis:
                return None
            try:
                result = await self.query_analysis_client.analyze_query(
                    query=query,
                    correlation_id=correlation_id
                )
                logger.info(f"[{correlation_id}] Query analysis completed (parallel)")
                return result
            except Exception as e:
                logger.warning(f"[{correlation_id}] Query analysis failed: {e}")
                return None

        async def do_retrieval():
            result = await self.document_retrieval_client.post(
                "/api/v1/retrieve",
                json={
                    "query": query,
                    "collection": collection,
                    "top_k": retrieval_top_k,
                    "score_threshold": score_threshold,
                    "use_hyde_colbert": use_hyde_colbert,
                    "hyde_colbert_options": hyde_colbert_options,
                },
                correlation_id=correlation_id
            )
            logger.info(f"[{correlation_id}] Document retrieval completed (parallel)")
            return result

        # Run both in parallel
        analysis_result, retrieval_result = await asyncio.gather(
            do_analysis(),
            do_retrieval(),
            return_exceptions=False
        )

        return analysis_result, retrieval_result

    async def execute_rag_pipeline(
        self,
        query: str,
        retrieval_top_k: int = 20,
        ranking_top_k: int = 10,
        output_format: str = "markdown",
        enable_query_analysis: bool = True,
        enable_ranking: bool = True,
        enable_latex_parsing: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete RAG pipeline

        Args:
            query: User query
            retrieval_top_k: Number of documents to retrieve
            ranking_top_k: Number of documents to keep after ranking
            output_format: Output format for response
            enable_query_analysis: Whether to enable query analysis
            enable_ranking: Whether to enable document ranking
            enable_latex_parsing: Whether to enable LaTeX parsing

        Returns:
            Complete pipeline result with formatted response
        """
        # Generate correlation ID for distributed tracing
        correlation_id = self._generate_correlation_id()

        logger.info(
            f"Starting RAG pipeline for query: '{query[:100]}...' "
            f"(correlation_id={correlation_id})"
        )

        pipeline_result = {
            "correlation_id": correlation_id,
            "query": query,
            "steps": {},
            "errors": []
        }

        try:
            # Step 1: Query Analysis (optional)
            if enable_query_analysis:
                try:
                    logger.info(f"[{correlation_id}] Step 1: Query Analysis")
                    analysis_result = await self.query_analysis_client.analyze_query(
                        query=query,
                        correlation_id=correlation_id
                    )
                    pipeline_result["steps"]["query_analysis"] = analysis_result
                    logger.info(f"[{correlation_id}] Query analysis completed")
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Query analysis failed: {e}, continuing without it")
                    pipeline_result["errors"].append(f"Query analysis failed: {str(e)}")

            # Step 2: Document Retrieval
            logger.info(f"[{correlation_id}] Step 2: Document Retrieval (top_k={retrieval_top_k})")
            retrieval_result = await self.document_retrieval_client.search_documents(
                query=query,
                top_k=retrieval_top_k,
                correlation_id=correlation_id
            )
            pipeline_result["steps"]["document_retrieval"] = retrieval_result

            documents = retrieval_result.get("documents", [])
            logger.info(f"[{correlation_id}] Retrieved {len(documents)} documents")

            if not documents:
                logger.warning(f"[{correlation_id}] No documents retrieved, generating response without context")
                documents = []

            # Step 3: Document Ranking (optional)
            if enable_ranking and documents:
                try:
                    logger.info(f"[{correlation_id}] Step 3: Document Ranking (top_k={ranking_top_k})")
                    ranking_result = await self.document_ranking_client.rank_documents(
                        query=query,
                        documents=documents,
                        top_k=ranking_top_k,
                        correlation_id=correlation_id
                    )
                    pipeline_result["steps"]["document_ranking"] = ranking_result

                    # Use ranked documents
                    documents = ranking_result.get("ranked_documents", documents)
                    logger.info(f"[{correlation_id}] Ranked {len(documents)} documents")
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Ranking failed: {e}, using retrieval order")
                    pipeline_result["errors"].append(f"Ranking failed: {str(e)}")

            # Step 4: LaTeX Parsing (if detected)
            latex_parsed = False
            if enable_latex_parsing:
                # Check if documents contain LaTeX
                for doc in documents:
                    content = doc.get("content", "")
                    if self._detect_latex(content):
                        try:
                            logger.info(f"[{correlation_id}] Step 4: LaTeX parsing detected in documents")
                            # Parse LaTeX in document content
                            # Note: In production, you might want to parse each formula individually
                            # For now, we just log that LaTeX was detected
                            latex_parsed = True
                            pipeline_result["steps"]["latex_detected"] = True
                            break
                        except Exception as e:
                            logger.warning(f"[{correlation_id}] LaTeX parsing failed: {e}")
                            pipeline_result["errors"].append(f"LaTeX parsing failed: {str(e)}")

            # Step 5: LLM Generation
            logger.info(f"[{correlation_id}] Step 5: LLM Generation")
            generation_result = await self.llm_generation_client.generate_response(
                query=query,
                context_documents=documents,
                correlation_id=correlation_id
            )
            pipeline_result["steps"]["llm_generation"] = generation_result

            generated_text = generation_result.get("generated_text", "")
            logger.info(
                f"[{correlation_id}] Generated response ({generation_result.get('tokens_used', 0)} tokens)"
            )

            # Prepare sources for formatting with URLs and metadata
            sources = []
            for doc in documents:
                metadata = doc.get("metadata", {})

                source = {
                    "title": metadata.get("title") or doc.get("title") or "Untitled Document",
                    "content": doc.get("content", "")[:300],  # First 300 chars as preview
                    "score": doc.get("relevance_score") or doc.get("score"),
                }

                # Extract URL from metadata (check common metadata fields)
                url = (
                    metadata.get("url") or
                    metadata.get("source_url") or
                    metadata.get("link") or
                    metadata.get("uri") or
                    metadata.get("document_url")
                )
                if url:
                    source["url"] = url

                # Add document ID if available
                doc_id = metadata.get("document_id") or metadata.get("id") or metadata.get("_id")
                if doc_id:
                    source["document_id"] = str(doc_id)

                # Add document type if available (e.g., "pdf", "webpage", "arxiv")
                doc_type = metadata.get("document_type") or metadata.get("type") or metadata.get("source_type")
                if doc_type:
                    source["document_type"] = doc_type

                # Add author if available
                author = metadata.get("author") or metadata.get("authors")
                if author:
                    source["author"] = author

                # Add publication date if available
                pub_date = metadata.get("publication_date") or metadata.get("date") or metadata.get("published")
                if pub_date:
                    source["publication_date"] = str(pub_date)

                sources.append(source)

            # Step 6: Response Formatting
            logger.info(f"[{correlation_id}] Step 6: Response Formatting (format={output_format})")
            format_result = await self.response_formatter_client.format_response(
                content=generated_text,
                query=query,
                sources=sources,
                output_format=output_format,
                include_citations=True,
                correlation_id=correlation_id
            )
            pipeline_result["steps"]["response_formatter"] = format_result

            logger.info(f"[{correlation_id}] RAG pipeline completed successfully")

            # Build final response
            return {
                "success": True,
                "correlation_id": correlation_id,
                "query": query,
                "response": format_result.get("formatted_content"),
                "output_format": output_format,
                "sources": sources,
                "metadata": {
                    "documents_retrieved": len(retrieval_result.get("documents", [])),
                    "documents_used": len(documents),
                    "tokens_used": generation_result.get("tokens_used"),
                    "model_used": generation_result.get("model_used"),
                    "latex_detected": latex_parsed,
                    "errors": pipeline_result["errors"] if pipeline_result["errors"] else None
                }
            }

        except Exception as e:
            logger.error(f"[{correlation_id}] RAG pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "correlation_id": correlation_id,
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__,
                "pipeline_steps_completed": list(pipeline_result["steps"].keys())
            }

    async def execute_rag_pipeline_stream(
        self,
        query: str,
        collection: str = "wikipedia",
        retrieval_top_k: int = 10,
        ranking_top_k: int = 5,
        score_threshold: float = 0.3,
        use_hyde_colbert: bool = False,
        hyde_colbert_options: Optional[Dict[str, Any]] = None,
        enable_query_analysis: bool = True,
        enable_ranking: bool = True,
        model: str = "llama3.2:1b"
    ):
        """
        Execute RAG pipeline with streaming LLM response.

        Yields SSE-formatted events:
        - {"type": "sources", "sources": [...]}
        - {"type": "token", "content": "..."}
        - {"type": "done"}
        - {"type": "error", "error": "..."}
        """
        import json as json_module

        correlation_id = self._generate_correlation_id()
        logger.info(f"[{correlation_id}] Starting streaming RAG pipeline for: {query[:50]}...")

        try:
            # Step 1 & 2: Run Query Analysis and Document Retrieval in PARALLEL for performance
            logger.info(f"[{correlation_id}] Steps 1-2: Parallel Query Analysis + Document Retrieval")

            try:
                analysis_result, retrieval_result = await self._parallel_query_analysis_and_retrieval(
                    query=query,
                    collection=collection,
                    retrieval_top_k=retrieval_top_k,
                    score_threshold=score_threshold,
                    use_hyde_colbert=use_hyde_colbert,
                    hyde_colbert_options=hyde_colbert_options,
                    enable_query_analysis=enable_query_analysis,
                    correlation_id=correlation_id
                )
            except Exception as e:
                logger.error(f"[{correlation_id}] Document retrieval failed: {e}")
                yield f"data: {json_module.dumps({'type': 'error', 'error': f'Retrieval failed: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Process analysis results for ranking
            analyzed_query = query
            latex_info = None
            if analysis_result:
                analysis = analysis_result.get("analysis", {})

                # Check if LaTeX was detected
                latex_analysis = analysis.get("latex_analysis")
                if latex_analysis and latex_analysis.get("has_latex"):
                    latex_info = {
                        "has_latex": True,
                        "formulas": latex_analysis.get("formulas", []),
                        "query_type": latex_analysis.get("query_type", "general"),
                        "search_queries": latex_analysis.get("search_queries", [])
                    }
                    # Use LaTeX-optimized query for ranking
                    search_queries = latex_info.get("search_queries", [])
                    if search_queries:
                        analyzed_query = search_queries[0]
                    logger.info(f"[{correlation_id}] LaTeX detected: {len(latex_info.get('formulas', []))} formulas")
                else:
                    # Use normalized query for ranking
                    if analysis.get("normalized_query"):
                        analyzed_query = analysis["normalized_query"]

            # Get documents from retrieval
            documents = retrieval_result.get("documents", [])
            logger.info(f"[{correlation_id}] Retrieved {len(documents)} documents")

            if not documents:
                yield f"data: {json_module.dumps({'type': 'error', 'error': 'No documents found'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Step 3: Document Ranking (optional)
            if enable_ranking and len(documents) > ranking_top_k:
                try:
                    logger.info(f"[{correlation_id}] Step 3: Document Ranking (top_k={ranking_top_k})")
                    ranking_result = await self.document_ranking_client.rank_documents(
                        query=analyzed_query,
                        documents=[
                            {"content": d.get("content", ""), "metadata": d.get("metadata", {})}
                            for d in documents
                        ],
                        top_k=ranking_top_k,
                        correlation_id=correlation_id
                    )
                    ranked_docs = ranking_result.get("ranked_documents", [])
                    if ranked_docs:
                        documents = ranked_docs
                    logger.info(f"[{correlation_id}] Ranking completed, {len(documents)} documents kept")
                except Exception as e:
                    logger.warning(f"[{correlation_id}] Ranking failed: {e}, using original order")

            # Prepare sources for response
            sources = []
            for idx, doc in enumerate(documents):
                metadata = doc.get("metadata", {})
                source = {
                    "title": metadata.get("title") or doc.get("title") or f"Source {idx + 1}",
                    "content": doc.get("content", "")[:500],
                    "score": doc.get("score") or doc.get("relevance_score", 0),
                    "metadata": metadata,
                }
                sources.append(source)

            # Send sources first
            yield f"data: {json_module.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Step 4: LLM Generation (streaming)
            logger.info(f"[{correlation_id}] Step 4: Streaming LLM Generation (model={model})")
            context_documents = [
                {
                    "content": doc.get("content", ""),
                    "title": doc.get("metadata", {}).get("title") or doc.get("title"),
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                }
                for doc in documents
            ]

            async for chunk in self.llm_generation_client.generate_stream(
                query=query,
                context_documents=context_documents,
                parameters={"model": model},
                correlation_id=correlation_id
            ):
                if "content" in chunk:
                    yield f"data: {json_module.dumps({'type': 'token', 'content': chunk['content']})}\n\n"

            yield "data: [DONE]\n\n"
            logger.info(f"[{correlation_id}] Streaming RAG pipeline completed")

        except Exception as e:
            logger.error(f"[{correlation_id}] Streaming pipeline error: {e}", exc_info=True)
            yield f"data: {json_module.dumps({'type': 'error', 'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    async def health_check_all_services(self) -> Dict[str, bool]:
        """
        Check health of all services

        Returns:
            Dictionary with service health status
        """
        services = {
            "query_analysis": self.query_analysis_client,
            "document_retrieval": self.document_retrieval_client,
            "document_ranking": self.document_ranking_client,
            "latex_parser": self.latex_parser_client,
            "llm_generation": self.llm_generation_client,
            "response_formatter": self.response_formatter_client,
        }

        health_status = {}
        for service_name, client in services.items():
            try:
                is_healthy = await client.health_check()
                health_status[service_name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                health_status[service_name] = False

        return health_status

    async def close(self):
        """Close all service clients"""
        await self.query_analysis_client.close()
        await self.document_retrieval_client.close()
        await self.document_ranking_client.close()
        await self.latex_parser_client.close()
        await self.llm_generation_client.close()
        await self.response_formatter_client.close()
        logger.info("All service clients closed")
