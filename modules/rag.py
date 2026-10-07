"""
modules/rag.py

Embeddings, vector store, retriever, and LLM wiring for the Laptop RAG app.

Laptop-oriented defaults:
- Embeddings: sentence-transformers "all-MiniLM-L6-v2" — small (~80MB),
  fast on CPU, good enough quality for local document search.
- Vector store: Chroma, persisted to disk so you don't re-embed on every run.
- LLM: Ollama by default (fully local, works with quantized GGUF models
  such as llama3.1:8b-instruct-q4_K_M or mistral:7b-instruct-q4_K_M), with
  an optional OpenAI backend for people who'd rather use an API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "laptop_rag_docs"

RAG_SYSTEM_PROMPT = (
    "You are Clippy, a friendly desktop assistant chatting with the user "
    "about their local documents. Reply naturally and briefly, like normal "
    "conversation. Use the provided context to answer questions about the "
    "documents and name the source file you used. For small talk, greetings, "
    "or anything the context doesn't actually help with, just answer "
    "conversationally — don't force the context in or claim uncertainty."
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """A lightweight local embedding model — runs on CPU, no API key,
    no GPU needed. all-MiniLM-L6-v2 is ~80MB and embeds a few hundred
    chunks per second on a typical laptop CPU.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # enables cosine similarity
    )


def build_vector_store(
    chunks: list[Document],
    persist_directory: str,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Embed chunks and persist them to a local Chroma store on disk."""
    embeddings = embeddings or get_embeddings()
    os.makedirs(persist_directory, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    store.persist()
    return store


def load_vector_store(
    persist_directory: str,
    embeddings: HuggingFaceEmbeddings | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Reopen a previously-persisted Chroma store without re-embedding."""
    embeddings = embeddings or get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def get_retriever(store: Chroma, k: int = 4):
    """Cosine-similarity retriever over the local vector store.

    Cosine similarity is enabled by normalizing embeddings at encode time
    (see get_embeddings) combined with Chroma's default 'l2' space on
    normalized vectors, which ranks identically to cosine similarity.
    """
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


@dataclass
class LLMConfig:
    backend: Literal["ollama", "openai"] = "ollama"
    model: str = "llama3.2:3b-instruct-q4_K_M"  # quantized GGUF via Ollama by default
    temperature: float = 0.6
    openai_api_key: str | None = None


def get_llm(config: LLMConfig):
    """Return a chat model for either a local Ollama backend (default,
    recommended for laptops — use a 4-bit/8-bit quantized GGUF model to
    minimize RAM use) or a hosted OpenAI-compatible API.
    """
    if config.backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.model, temperature=config.temperature)
    elif config.backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Export it or pass openai_api_key explicitly."
            )
        return ChatOpenAI(model=config.model, temperature=config.temperature, api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM backend: {config.backend}")


def _format_docs(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarizes the user's local documents "
    "using only the provided excerpts. Write a concise summary as a short "
    "list of key points. If a topic is given, focus the summary on that "
    "topic and say if the excerpts don't cover it. Cite source file "
    "name(s) you drew from at the end."
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM_PROMPT),
        (
            "human",
            "Excerpts:\n{context}\n\n"
            "{instruction}",
        ),
    ]
)


def summarize_with_sources(
    store: Chroma,
    llm_config: LLMConfig,
    topic: str | None = None,
    k: int = 8,
):
    """Summarize the indexed documents, optionally focused on a topic.

    Reuses similarity search against the topic (or a generic overview
    query when no topic is given) to pull a representative sample of
    chunks, then asks the LLM to summarize just those excerpts. Returns
    (summary: str, docs: list[Document]) like query_with_sources.
    """
    search_query = topic or "overview of the main topics and key information"
    docs = get_retriever(store, k=k).invoke(search_query)
    llm = get_llm(llm_config)

    instruction = (
        f"Summarize what these excerpts say about: {topic}"
        if topic
        else "Summarize the key points across these excerpts."
    )
    messages = SUMMARY_PROMPT.invoke({"context": _format_docs(docs), "instruction": instruction})
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, docs


CHAT_SYSTEM_PROMPT = (
    "You are Clippy, a friendly desktop assistant. Reply naturally and "
    "briefly to conversational messages like greetings or thanks. You "
    "also have access to the user's local documents for questions about "
    "their files, but this particular message doesn't need that."
)


def chat_reply(llm_config: LLMConfig, message: str) -> str:
    """Plain conversational reply with no document retrieval - for small
    talk that doesn't need (and shouldn't be forced through) RAG context."""
    llm = get_llm(llm_config)
    response = llm.invoke([("system", CHAT_SYSTEM_PROMPT), ("human", message)])
    return response.content if hasattr(response, "content") else str(response)


def query_with_sources(store: Chroma, llm_config: LLMConfig, question: str, k: int = 4, history: list[tuple[str, str]] | None = None):
    """Retrieve top-k chunks for the question and ask the LLM to answer,
    optionally incorporating chat history. Returns (answer, docs)."""
    retriever = get_retriever(store, k=k)
    docs = retriever.invoke(question)
    llm = get_llm(llm_config)

    # Format history if available
    history_text = ""
    if history:
        history_lines = []
        for q, a in history[-5:]:  # Limit to last 5 turns to save context window
            history_lines.append(f"User: {q}\nAssistant: {a}")
        history_text = "\n".join(history_lines) + "\n\n"

    context_str = _format_docs(docs)
    full_context = f"{history_text}Context:\n{context_str}" if history_text else f"Context:\n{context_str}"

    messages = RAG_PROMPT.invoke({"context": full_context, "question": question})
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, docs