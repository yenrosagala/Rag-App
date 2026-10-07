"""
modules/engine.py

Thin wrapper around ingest.py / rag.py that:
- Persists user settings (data folder, backend, model, k) to settings.json
- Lazily loads / rebuilds the Chroma vector store
- Exposes plain blocking functions (rebuild_index, ask) meant to be called
  from a background thread by the Tkinter UI — nothing here touches Tk.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict

from .ingest import ingest_folder, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SUPPORTED_EXTENSIONS
from .rag import (
    LLMConfig,
    build_vector_store,
    load_vector_store,
    get_embeddings,
    query_with_sources,
    summarize_with_sources,
    chat_reply,
)


def _base_dir() -> str:
    """Folder settings.json/data/vector_store live next to.

    When frozen by PyInstaller (--onefile), the app runs its code from a
    temp extraction folder that's deleted on exit, so __file__ can't be
    used to find "the project folder" anymore - there isn't one. In that
    case, anchor to the folder the .exe itself sits in instead, so state
    persists across runs and survives you deleting the source files.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = _base_dir()
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")


@dataclass
class Settings:
    data_dir: str = DEFAULT_DATA_DIR
    vector_store_dir: str = DEFAULT_VECTOR_STORE_DIR
    backend: str = "ollama"
    model: str = "llama3:8b-instruct-q8_0"
    openai_api_key: str = ""
    k: int = 4
    avatar_style: str = "classic"
    avatar_shape: str = "paperclip"

    @classmethod
    def load(cls) -> "Settings":
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                known = {k: v for k, v in data.items() if k in asdict(cls())}
                return cls(**{**asdict(cls()), **known})
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def llm_config(self) -> LLMConfig:
        return LLMConfig(
            backend=self.backend,
            model=self.model,
            openai_api_key=self.openai_api_key or None,
        )


class RagEngine:
    """Holds the loaded vector store and settings for the app's lifetime.

    All methods here are synchronous/blocking by design — the caller
    (the Tkinter UI) is responsible for running them off the main thread.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self._store = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def is_indexed(self) -> bool:
        d = self.settings.vector_store_dir
        return os.path.isdir(d) and bool(os.listdir(d))

    def rebuild_index(self, progress_cb=None):
        """Ingest + chunk + embed everything in data_dir, persist to Chroma.

        progress_cb(str) is called with human-readable status updates.
        Returns an IngestStats-like object.
        """
        def report(msg):
            if progress_cb:
                progress_cb(msg)

        report("Scanning documents...")
        chunks, stats = ingest_folder(
            self.settings.data_dir,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError(
                f"No supported documents found in {self.settings.data_dir} "
                "(looking for .pdf, .txt, .md)."
            )

        report(f"Embedding {len(chunks)} chunk(s) locally... (this can take a bit on CPU)")
        embeddings = self._get_embeddings()
        self._store = build_vector_store(
            chunks, self.settings.vector_store_dir, embeddings=embeddings
        )
        report(
            f"Indexed {stats.files_loaded}/{stats.files_seen} file(s) "
            f"into {stats.chunks_created} chunk(s)."
        )
        return stats

    def _get_store(self):
        if self._store is not None:
            return self._store
        if self.is_indexed():
            self._store = load_vector_store(
                self.settings.vector_store_dir, embeddings=self._get_embeddings()
            )
            return self._store
        return None

    def ask(self, question: str, history: list[tuple[str, str]] | None = None):
        """Returns (answer: str, sources: list[Document]).

        If no index exists yet, falls back to a plain conversational reply
        (sources=[]) instead of erroring - there's nothing to retrieve from
        either way, so a greeting or a real question gets a normal answer
        rather than a "no index yet" error. Once an index exists, retrieval
        always runs and the LLM itself (via the system prompt) decides
        whether the retrieved context is actually relevant to the message.
        """
        store = self._get_store()
        if store is None:
            return self.chat(question), []
        answer, docs = query_with_sources(
            store, self.settings.llm_config(), question, k=self.settings.k, history=history
        )
        return answer, docs

    def summarize(self, topic: str | None = None):
        """Returns (summary: str, sources: list[Document]).

        Like ask(), but pulls a wider sample of chunks (topic-focused if
        given, otherwise a general overview query) and asks the LLM to
        summarize them instead of answering a specific question.

        Raises RuntimeError if no index exists yet.
        """
        store = self._get_store()
        if store is None:
            raise RuntimeError(
                "No index yet — open Settings and click 'Rebuild index' first."
            )
        k = max(self.settings.k * 2, 8)
        return summarize_with_sources(store, self.settings.llm_config(), topic=topic, k=k)

    def chat(self, message: str) -> str:
        """Plain conversational reply, no document retrieval - doesn't
        require an index to exist yet."""
        return chat_reply(self.settings.llm_config(), message)

    def list_source_files(self) -> list[str]:
        """Filenames under data_dir that are eligible for indexing (no LLM
        or vector store needed — just a folder scan)."""
        data_dir = self.settings.data_dir
        if not os.path.isdir(data_dir):
            return []
        found = []
        for root, _dirs, files in os.walk(data_dir):
            for name in files:
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                    found.append(os.path.relpath(os.path.join(root, name), data_dir))
        return sorted(found)
