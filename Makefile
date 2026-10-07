.PHONY: setup run build clean

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete!"
	@echo "Next: install Ollama (https://ollama.com), run 'ollama pull llama3:8b-instruct-q8_0',"
	@echo "add documents to data/, then run 'make run'."

run:
	$(PY) clippy_app.py

build:
	$(VENV)/bin/pyinstaller --noconfirm --onefile --windowed --name Clippy \
	    --add-data "modules:modules" \
	    --add-data "assets:assets" \
	    --collect-all sentence_transformers \
	    --collect-all chromadb \
	    clippy_app.py

clean:
	rm -rf $(VENV) vector_store/* build dist Clippy.spec
