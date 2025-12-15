"""
Prompt templates for LLM generation
"""
from typing import List, Dict, Any


class PromptTemplates:
    """Prompt templates for different query types and contexts"""

    SYSTEM_PROMPT = """You are a helpful AI assistant that provides accurate, well-structured answers based on the given context.

Your responsibilities:
1. Answer questions using ONLY the information provided in the context documents
2. If the context doesn't contain enough information, clearly state that
3. Cite sources when making specific claims
4. Provide clear, concise, and well-organized responses
5. Use markdown formatting for better readability
6. For mathematical content, use LaTeX with dollar sign delimiters:
   - Use $...$ for inline math (e.g., $x = 5$, $2x + 3$)
   - Use $$...$$ for display/block equations (e.g., $$x = \\frac{10}{2}$$)
   - NEVER use \\(...\\) or \\[...\\] delimiters

Important guidelines:
- Never make up information not present in the context
- Be honest about limitations in the available information
- Structure longer responses with headings and bullet points
- Maintain a professional and helpful tone
- Always use $ and $$ delimiters for LaTeX math, never parentheses or brackets"""

    RAG_PROMPT_TEMPLATE = """Based on the following context documents, please answer the user's query.

Context Documents:
{context}

User Query: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to fully answer the query, acknowledge this limitation."""

    NO_CONTEXT_PROMPT_TEMPLATE = """The user has asked the following query, but no relevant context documents were found:

User Query: {query}

Please provide a helpful response explaining that you don't have specific information to answer this query based on the available documents."""

    CITATION_TEMPLATE = "[Source: {source}]"

    @staticmethod
    def format_context_documents(documents: List[Dict[str, Any]]) -> str:
        """
        Format context documents into a readable string

        Args:
            documents: List of document dictionaries

        Returns:
            Formatted context string
        """
        if not documents:
            return "No context documents provided."

        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            doc_text = f"Document {idx}"

            # Add title if available
            if doc.get('title'):
                doc_text += f": {doc['title']}"

            doc_text += "\n"

            # Add source if available
            if doc.get('source'):
                doc_text += f"Source: {doc['source']}\n"

            # Add relevance score if available
            if doc.get('score') is not None:
                doc_text += f"Relevance Score: {doc['score']:.3f}\n"

            # Add content
            doc_text += f"Content:\n{doc['content']}\n"

            formatted_docs.append(doc_text)

        return "\n" + "-" * 80 + "\n".join(formatted_docs)

    @staticmethod
    def build_rag_prompt(query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build the complete RAG prompt with context

        Args:
            query: User query
            documents: List of context documents

        Returns:
            Complete prompt string
        """
        if not documents:
            return PromptTemplates.NO_CONTEXT_PROMPT_TEMPLATE.format(query=query)

        context = PromptTemplates.format_context_documents(documents)
        return PromptTemplates.RAG_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

    @staticmethod
    def extract_sources(documents: List[Dict[str, Any]]) -> List[str]:
        """
        Extract source information from documents

        Args:
            documents: List of context documents

        Returns:
            List of source strings
        """
        sources = []
        for doc in documents:
            source = doc.get('source')
            if source:
                sources.append(source)
            elif doc.get('title'):
                sources.append(doc['title'])

        return list(set(sources))  # Remove duplicates
