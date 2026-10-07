"""
modules/ingest.py

Document loading, parsing, and chunking for the Laptop RAG app.

Design goals (laptop-friendly):
- No heavy parsing dependencies beyond pypdf.
- Deterministic, resumable: re-ingesting the same folder is cheap and safe.
- Chunking uses RecursiveCharacterTextSplitter (chunk_size=500, overlap=50),
  which keeps context windows small so a local embedding model / small LLM
  can work with them comfortably on CPU.
"""
from __future__ import annotations

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Most of the wall-clock cost of loading a folder (especially a cloud-synced
# Drive folder, where a file may need to be pulled down on first touch) is
# I/O, not CPU, so a small thread pool speeds this up even under the GIL.
MAX_LOAD_WORKERS = min(8, (os.cpu_count() or 4) * 2)


@dataclass
class IngestStats:
    files_seen: int = 0
    files_loaded: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    errors: list[str] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _iter_source_files(data_dir: str) -> Iterable[str]:
    """Walk data_dir and yield paths to files we know how to load."""
    for root, _dirs, files in os.walk(data_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                yield os.path.join(root, name)


def _load_single_file(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    else:
        # .txt / .md — try utf-8 first, fall back to autodetect
        loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def load_documents(
    data_dir: str,
    stats: IngestStats | None = None,
) -> list[Document]:
    """Load supported documents under data_dir into LangChain Documents.

    Files are loaded concurrently with a small thread pool since the
    dominant cost is I/O, not CPU.

    Individual file failures are logged and skipped rather than aborting
    the whole ingestion run — a single corrupt PDF shouldn't block the rest
    of the folder.
    """
    stats = stats or IngestStats()
    documents: list[Document] = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    paths = list(_iter_source_files(data_dir))
    stats.files_seen += len(paths)

    with ThreadPoolExecutor(max_workers=MAX_LOAD_WORKERS) as pool:
        future_to_path = {pool.submit(_load_single_file, p): p for p in paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                docs = future.result()
                for d in docs:
                    d.metadata["source"] = os.path.relpath(path, data_dir)
                documents.extend(docs)
                stats.files_loaded += 1
            except Exception as exc:  # noqa: BLE001 - we want to keep going
                stats.files_skipped += 1
                msg = f"Skipped '{path}': {exc}"
                logger.warning(msg)
                stats.errors.append(msg)

    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks using recursive character
    splitting. Small chunk sizes (default 500/50) keep retrieval precise
    and keep prompts short enough for local models on modest hardware.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_folder(
    data_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[Document], IngestStats]:
    """High-level convenience entrypoint: load + chunk everything in data_dir."""
    # ponytail: full re-ingest of the folder on every rebuild — add an
    # mtime/size manifest diff (scan_files/diff_manifest + Chroma upsert)
    # only if rebuild time actually hurts.
    stats = IngestStats()
    raw_docs = load_documents(data_dir, stats=stats)
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats.chunks_created = len(chunks)
    return chunks, stats