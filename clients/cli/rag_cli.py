#!/usr/bin/env python3
"""
RAG System CLI Client
Full-featured command-line interface for the RAG system.
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

console = Console()


@dataclass
class Settings:
    """CLI Settings"""
    api_url: str = "http://localhost:8000/api/v1"
    collection: str = ""
    top_k: int = 5
    score_threshold: float = 0.3
    retrieval_method: str = "hybrid"  # hybrid, dense, hyde_colbert, rag_fusion
    enable_query_analysis: bool = True
    enable_ranking: bool = True
    model: str = "llama3.2:3b"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Hybrid options
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    fusion_method: str = "weighted"  # weighted, rrf

    # HyDE-ColBERT options
    n_hypotheticals: int = 3
    domain: str = "general"
    fusion_strategy: str = "max_score"
    fusion_weight: float = 0.6

    def to_dict(self) -> dict:
        return {
            "api_url": self.api_url,
            "collection": self.collection,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "retrieval_method": self.retrieval_method,
            "enable_query_analysis": self.enable_query_analysis,
            "enable_ranking": self.enable_ranking,
            "model": self.model,
        }


@dataclass
class ChatMessage:
    """Chat message"""
    role: str  # user, assistant
    content: str
    sources: list = field(default_factory=list)


class RAGClient:
    """RAG API Client"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = await self.client.get(f"{self.settings.api_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_collections(self) -> list:
        """Get available collections"""
        try:
            response = await self.client.get(f"{self.settings.api_url}/collections")
            if response.status_code == 200:
                data = response.json()
                return data.get("collections", [])
        except Exception as e:
            console.print(f"[red]Error fetching collections: {e}[/red]")
        return []

    async def upload_document(self, file_path: str, collection_name: Optional[str] = None) -> dict:
        """Upload a document"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        collection = collection_name or self.settings.collection

        with open(path, "rb") as f:
            files = {"file": (path.name, f, self._get_mime_type(path))}
            data = {"session_id": self.settings.session_id}
            if collection:
                data["collection_name"] = collection

            response = await self.client.post(
                f"{self.settings.api_url}/documents/upload",
                files=files,
                data=data,
            )

        if response.status_code != 200:
            error = response.json().get("detail", "Upload failed")
            raise Exception(error)

        return response.json()

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for file"""
        suffix = path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }
        return mime_types.get(suffix, "application/octet-stream")

    async def query_stream(self, query: str) -> AsyncGenerator[dict, None]:
        """Stream a query response"""
        request_body = {
            "query": query,
            "collection": self.settings.collection,
            "top_k": self.settings.top_k,
            "score_threshold": self.settings.score_threshold,
            "model": self.settings.model,
            "enable_query_analysis": self.settings.enable_query_analysis,
            "enable_ranking": self.settings.enable_ranking,
            "retrieval_method": self.settings.retrieval_method,
        }

        # Add method-specific options
        if self.settings.retrieval_method == "hybrid":
            request_body["hybrid_options"] = {
                "dense_weight": self.settings.dense_weight,
                "sparse_weight": self.settings.sparse_weight,
                "fusion_method": self.settings.fusion_method,
            }
        elif self.settings.retrieval_method == "hyde_colbert":
            request_body["use_hyde_colbert"] = True
            request_body["hyde_colbert_options"] = {
                "n_hypotheticals": self.settings.n_hypotheticals,
                "domain": self.settings.domain,
                "fusion_strategy": self.settings.fusion_strategy,
                "fusion_weight": self.settings.fusion_weight,
            }

        async with self.client.stream(
            "POST",
            f"{self.settings.api_url}/query/stream",
            json=request_body,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                yield {"type": "error", "error": f"HTTP {response.status_code}"}
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"type": "done"}
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass


class RAGCLI:
    """Main CLI Application"""

    def __init__(self):
        self.settings = Settings()
        self.client = RAGClient(self.settings)
        self.chat_history: list[ChatMessage] = []
        self.config_path = Path.home() / ".rag_cli_config.json"
        self._load_config()

    def _load_config(self):
        """Load saved config"""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                for key, value in config.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
            except Exception:
                pass

    def _save_config(self):
        """Save config"""
        config = {
            "api_url": self.settings.api_url,
            "collection": self.settings.collection,
            "top_k": self.settings.top_k,
            "score_threshold": self.settings.score_threshold,
            "retrieval_method": self.settings.retrieval_method,
            "enable_query_analysis": self.settings.enable_query_analysis,
            "enable_ranking": self.settings.enable_ranking,
            "model": self.settings.model,
            "dense_weight": self.settings.dense_weight,
            "sparse_weight": self.settings.sparse_weight,
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def print_banner(self):
        """Print welcome banner"""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════╗
║                    RAG System CLI                          ║
║         Interactive Document Q&A with AI                   ║
╚═══════════════════════════════════════════════════════════╝[/bold cyan]
        """
        console.print(banner)
        console.print("[dim]Type /help for available commands[/dim]\n")

    def print_help(self):
        """Print help message"""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description")

        commands = [
            ("/help, /h", "Show this help message"),
            ("/collections, /c", "List available collections"),
            ("/select <name>", "Select a collection"),
            ("/upload <path>", "Upload a document"),
            ("/settings, /s", "Show/modify settings"),
            ("/history", "Show chat history"),
            ("/clear", "Clear chat history"),
            ("/sources", "Show sources from last response"),
            ("/quit, /q, /exit", "Exit the CLI"),
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        console.print(help_table)
        console.print("\n[dim]Or just type your question to chat![/dim]\n")

    async def list_collections(self):
        """List available collections"""
        with console.status("[bold green]Fetching collections..."):
            collections = await self.client.get_collections()

        if not collections:
            console.print("[yellow]No collections found. Upload a document first.[/yellow]")
            return

        table = Table(title="Available Collections", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Vectors", justify="right")
        table.add_column("Status")

        for col in collections:
            name = col.get("name", "Unknown")
            doc_count = col.get("documentCount", col.get("vectors_count", 0))
            is_selected = " [bold magenta](selected)[/bold magenta]" if name == self.settings.collection else ""
            table.add_row(f"{name}{is_selected}", str(doc_count), "[green]●[/green]")

        console.print(table)

    async def select_collection(self, name: str):
        """Select a collection"""
        collections = await self.client.get_collections()
        collection_names = [c.get("name") for c in collections]

        if name not in collection_names:
            console.print(f"[red]Collection '{name}' not found.[/red]")
            console.print(f"[dim]Available: {', '.join(collection_names)}[/dim]")
            return

        self.settings.collection = name
        self._save_config()
        console.print(f"[green]Selected collection: {name}[/green]")

    async def upload_document(self, file_path: str):
        """Upload a document"""
        path = Path(file_path).expanduser()
        if not path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Uploading {path.name}...", total=None)
            try:
                result = await self.client.upload_document(str(path))
                progress.update(task, completed=True)

                console.print(f"[green]✓ Uploaded successfully![/green]")
                console.print(f"  Collection: [cyan]{result.get('collection', 'default')}[/cyan]")
                console.print(f"  Chunks created: [cyan]{result.get('chunks_created', 'N/A')}[/cyan]")

                # Auto-select collection if not set
                if not self.settings.collection and result.get("collection"):
                    self.settings.collection = result["collection"]
                    self._save_config()
                    console.print(f"  [dim]Auto-selected collection: {result['collection']}[/dim]")
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Upload failed: {e}[/red]")

    def show_settings(self):
        """Show current settings"""
        table = Table(title="Current Settings", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        settings = [
            ("API URL", self.settings.api_url),
            ("Collection", self.settings.collection or "[dim]not set[/dim]"),
            ("Model", self.settings.model),
            ("Top K", str(self.settings.top_k)),
            ("Score Threshold", str(self.settings.score_threshold)),
            ("Retrieval Method", self.settings.retrieval_method),
            ("Query Analysis", "✓" if self.settings.enable_query_analysis else "✗"),
            ("Ranking", "✓" if self.settings.enable_ranking else "✗"),
        ]

        for name, value in settings:
            table.add_row(name, str(value))

        console.print(table)
        console.print("\n[dim]Use /settings <key> <value> to change a setting[/dim]")

    def modify_setting(self, key: str, value: str):
        """Modify a setting"""
        key = key.lower()

        setting_map = {
            "api_url": ("api_url", str),
            "url": ("api_url", str),
            "collection": ("collection", str),
            "model": ("model", str),
            "top_k": ("top_k", int),
            "topk": ("top_k", int),
            "k": ("top_k", int),
            "score_threshold": ("score_threshold", float),
            "threshold": ("score_threshold", float),
            "retrieval_method": ("retrieval_method", str),
            "method": ("retrieval_method", str),
            "query_analysis": ("enable_query_analysis", lambda x: x.lower() in ("true", "1", "yes", "on")),
            "analysis": ("enable_query_analysis", lambda x: x.lower() in ("true", "1", "yes", "on")),
            "ranking": ("enable_ranking", lambda x: x.lower() in ("true", "1", "yes", "on")),
        }

        if key not in setting_map:
            console.print(f"[red]Unknown setting: {key}[/red]")
            console.print(f"[dim]Available: {', '.join(set(k for k, _ in setting_map.values()))}[/dim]")
            return

        attr_name, converter = setting_map[key]

        # Validate retrieval method
        if attr_name == "retrieval_method":
            valid_methods = ["hybrid", "dense", "hyde_colbert", "rag_fusion"]
            if value not in valid_methods:
                console.print(f"[red]Invalid retrieval method: {value}[/red]")
                console.print(f"[dim]Valid methods: {', '.join(valid_methods)}[/dim]")
                return

        try:
            setattr(self.settings, attr_name, converter(value))
            self._save_config()
            console.print(f"[green]✓ Set {attr_name} = {value}[/green]")
        except Exception as e:
            console.print(f"[red]Invalid value: {e}[/red]")

    def show_history(self):
        """Show chat history"""
        if not self.chat_history:
            console.print("[dim]No chat history yet.[/dim]")
            return

        for i, msg in enumerate(self.chat_history):
            if msg.role == "user":
                console.print(f"\n[bold blue]You:[/bold blue] {msg.content}")
            else:
                console.print(f"\n[bold green]Assistant:[/bold green]")
                console.print(Markdown(msg.content))

    def show_last_sources(self):
        """Show sources from last response"""
        # Find last assistant message
        for msg in reversed(self.chat_history):
            if msg.role == "assistant" and msg.sources:
                self._display_sources(msg.sources)
                return
        console.print("[dim]No sources available from last response.[/dim]")

    def _display_sources(self, sources: list):
        """Display sources in a nice format"""
        if not sources:
            console.print("[dim]No sources found.[/dim]")
            return

        console.print("\n[bold cyan]Sources:[/bold cyan]")
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            title = src.get("title", "Unknown")
            content = src.get("content", "")[:200] + "..." if len(src.get("content", "")) > 200 else src.get("content", "")
            metadata = src.get("metadata", {})

            # Build metadata string
            meta_parts = []
            if metadata.get("source"):
                meta_parts.append(f"source: {metadata['source']}")
            if metadata.get("page"):
                meta_parts.append(f"page: {metadata['page']}")
            meta_str = " | ".join(meta_parts) if meta_parts else ""

            panel = Panel(
                f"[dim]{content}[/dim]\n\n[cyan]{meta_str}[/cyan]",
                title=f"[bold]{i}. {title}[/bold] [green](score: {score:.3f})[/green]",
                border_style="dim",
            )
            console.print(panel)

    async def chat(self, query: str):
        """Send a chat query and stream the response"""
        if not self.settings.collection:
            console.print("[yellow]No collection selected. Use /collections to list and /select to choose one.[/yellow]")
            return

        # Add user message to history
        self.chat_history.append(ChatMessage(role="user", content=query))

        response_text = ""
        sources = []

        console.print()

        with Live(console=console, refresh_per_second=10) as live:
            try:
                async for chunk in self.client.query_stream(query):
                    chunk_type = chunk.get("type")

                    if chunk_type == "token":
                        response_text += chunk.get("content", "")
                        live.update(Markdown(response_text + "▌"))

                    elif chunk_type == "sources":
                        sources = chunk.get("sources", [])

                    elif chunk_type == "error":
                        error = chunk.get("error", "Unknown error")
                        live.update(Text(f"[Error: {error}]", style="red"))
                        return

                    elif chunk_type == "done":
                        live.update(Markdown(response_text))
                        break

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                return

        # Add assistant response to history
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=response_text,
            sources=sources
        ))

        # Display sources
        if sources:
            self._display_sources(sources)

        console.print()

    async def run(self):
        """Main run loop"""
        self.print_banner()

        # Health check
        with console.status("[bold green]Connecting to API..."):
            healthy = await self.client.health_check()

        if not healthy:
            console.print(f"[red]Cannot connect to API at {self.settings.api_url}[/red]")
            console.print("[dim]Make sure the RAG system is running.[/dim]")
            return

        console.print(f"[green]✓ Connected to {self.settings.api_url}[/green]")

        # Show current collection
        if self.settings.collection:
            console.print(f"[dim]Current collection: {self.settings.collection}[/dim]")
        else:
            console.print("[yellow]No collection selected. Use /collections to list available collections.[/yellow]")

        console.print()

        # Main loop
        try:
            while True:
                try:
                    user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=2)
                    cmd = parts[0].lower()

                    if cmd in ("/quit", "/q", "/exit"):
                        console.print("[dim]Goodbye![/dim]")
                        break

                    elif cmd in ("/help", "/h"):
                        self.print_help()

                    elif cmd in ("/collections", "/c"):
                        await self.list_collections()

                    elif cmd == "/select":
                        if len(parts) < 2:
                            console.print("[red]Usage: /select <collection_name>[/red]")
                        else:
                            await self.select_collection(parts[1])

                    elif cmd == "/upload":
                        if len(parts) < 2:
                            console.print("[red]Usage: /upload <file_path>[/red]")
                        else:
                            await self.upload_document(parts[1])

                    elif cmd in ("/settings", "/s"):
                        if len(parts) >= 3:
                            self.modify_setting(parts[1], parts[2])
                        else:
                            self.show_settings()

                    elif cmd == "/history":
                        self.show_history()

                    elif cmd == "/clear":
                        self.chat_history.clear()
                        console.print("[dim]Chat history cleared.[/dim]")

                    elif cmd == "/sources":
                        self.show_last_sources()

                    else:
                        console.print(f"[red]Unknown command: {cmd}[/red]")
                        console.print("[dim]Type /help for available commands.[/dim]")

                else:
                    # Regular chat query
                    await self.chat(user_input)

        finally:
            await self.client.close()


async def main():
    """Entry point"""
    cli = RAGCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
