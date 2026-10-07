# Laptop RAG App — Clippy Edition

A local-first Retrieval-Augmented Generation (RAG) assistant for chatting
with your own documents — as a draggable Clippy-style desktop widget
instead of a browser tab. Built to run comfortably on a laptop CPU:

- **Tkinter** desktop widget UI (draggable, speech-bubble chat, no browser)
- **LangChain** for orchestration
- **sentence-transformers** (`all-MiniLM-L6-v2`) for fast local embeddings
- **Chroma** as a persistent local vector database
- **Ollama** for a fully local, quantized LLM (or optionally an OpenAI-compatible API)
- **PyInstaller** to package it into a single `Clippy.exe`

## 1. Prerequisites

- **Python 3.10+** (Tkinter ships with the standard installer on Windows/macOS;
  on Linux install it separately, e.g. `sudo apt install python3-tk`)
- **[Ollama](https://ollama.com)** — for fully local inference (recommended).
  After installing, pull a quantized model, e.g.:
  ```bash
  ollama pull llama3:8b-instruct-q8_0
  ```
  Quantized (4-bit/8-bit) GGUF models keep RAM usage low and run reasonably
  fast on CPU-only laptops. `mistral:7b-instruct-q4_K_M` is a good lighter
  alternative.
- *(Optional)* An **OpenAI API key**, if you'd rather use a hosted model
  instead of running one locally — entered directly in the app's Settings window.

## 2. Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or just run `build.bat` (Windows, also builds `Clippy.exe`) / `make setup` (macOS/Linux).

## 3. Add your documents

Drop `.pdf`, `.txt`, or `.md` files into the `data/` folder, or point Clippy
at any folder from the Settings window.

## 4. Run the app

```bash
python clippy_app.py       # or: make run
```

Clippy appears floating on your desktop:

1. **Drag** him anywhere by clicking and holding.
2. **Click** (no drag) to open his speech-bubble chat window.
3. **Right-click** for a menu with **Settings...** and **Quit**, or click
   the ⚙ icon inside the speech bubble.
4. In Settings: pick your documents folder, choose the LLM backend/model,
   and click **🔄 Rebuild index** — that runs ingestion, chunking, and local
   embedding into Chroma (`vector_store/`). Re-run any time your files change.
5. Ask questions in the speech bubble. Answers are grounded in retrieved
   document chunks, with source file names appended.

## Building a standalone `Clippy.exe`

```
build.bat
```

This sets up a venv, installs dependencies, and runs PyInstaller
(`--onefile --windowed`) to produce `dist\Clippy.exe` — a single file you
can copy anywhere alongside a `data\` folder. On macOS/Linux use
`make build` after `make setup` (adjust the `--add-data` separator to `:`,
already done in the Makefile).

## Project structure

```
laptop-rag-app/
├── clippy_app.py        # Tkinter Clippy widget UI (drag, speech bubble, settings)
├── requirements.txt      # Python dependencies
├── README.md              # This file
├── build.bat               # One-click Windows setup + PyInstaller build
├── Makefile                 # Cross-platform setup / run / build (macOS/Linux)
├── settings.json            # Auto-created on first Settings save (gitignored)
├── data/                     # Put your source documents here
├── vector_store/              # Persistent Chroma embeddings (auto-created)
└── modules/
    ├── ingest.py               # Document loading, parsing, chunking
    ├── rag.py                   # Embeddings, vector store, retriever, LLM, prompting
    └── engine.py                 # Settings persistence + blocking facade the UI drives
                                    #   from background threads
```

## How it works

1. **Ingest** (`modules/ingest.py`): loads PDFs/text files from `data/`,
   then splits them with `RecursiveCharacterTextSplitter` (chunk size 500,
   overlap 50) — small chunks keep retrieval precise and keep prompts short
   enough for a local model to handle well on modest hardware.
2. **Embed + store** (`modules/rag.py`): each chunk is embedded locally with
   `all-MiniLM-L6-v2` (normalized vectors, so similarity search is
   effectively cosine similarity) and persisted to a local Chroma database
   in `vector_store/`.
3. **Retrieve + answer**: on each question, the top-`k` most similar chunks
   are retrieved and passed to the LLM (Ollama by default) along with the
   question, and the model answers using only that context.
4. **UI** (`clippy_app.py` + `modules/engine.py`): the Tkinter widget never
   calls the RAG pipeline directly on the main thread — indexing and
   question-answering both run in background threads, with results handed
   back to the UI via a thread-safe queue polled with `root.after`, so
   Clippy stays draggable and responsive while a model "thinks."

## Switching LLM backends

Open Settings (right-click Clippy, or the ⚙ in the speech bubble):

- **Local (default):** backend `ollama`, enter the model name you pulled
  (e.g. `llama3:8b-instruct-q8_0`).
- **Hosted API:** backend `openai`, enter your API key and a model like
  `gpt-4o-mini`.

## Performance tips for laptops

- Prefer **quantized models** (4-bit/8-bit GGUF via Ollama) — they use far
  less RAM/VRAM than full-precision weights with a modest quality trade-off.
- `all-MiniLM-L6-v2` is intentionally small (~80 MB) so embedding runs
  quickly on a CPU without draining your battery.
- Keep chunk size modest (default 500 chars) — smaller prompts mean faster
  generation, especially on CPU-only inference.
- Re-index only when your documents change; embeddings are persisted to
  `vector_store/` so you don't pay the embedding cost on every launch.

## Troubleshooting

- **"No index yet" in the speech bubble:** open Settings and click
  **🔄 Rebuild index** first.
- **Error talking to Ollama / connection errors:** make sure `ollama serve`
  is running and that you've pulled the model named in Settings.
- **No documents found when indexing:** confirm files are directly inside
  `data/` (subfolders are also scanned) and have a `.pdf`, `.txt`, or `.md`
  extension.
- **Clippy has a solid pink-ish square background instead of being
  transparent:** `-transparentcolor` is a Windows-only Tk feature; on
  macOS/Linux he'll render on a plain background instead.
- **Slow indexing on large PDFs:** this is expected on CPU — consider
  splitting very large PDFs or indexing a smaller folder first.
