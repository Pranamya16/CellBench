#!/bin/sh
# Start the live demo: Ollama (Qwen) on this machine + the CellBench container. Open http://localhost:8000
set -e
curl -s -m 3 http://localhost:11434/api/tags >/dev/null || { echo "Ollama is not running. Start it (open the Ollama app, or run: ollama serve) and try again."; exit 1; }
ollama list | grep -q "qwen2.5:3b" || echo "Note: for the chat run: ollama pull qwen2.5:3b && ollama pull nomic-embed-text (the chat falls back to a weaker mode without them)"
ollama list | grep -q "qwen2.5:0.5b" || { echo "Model missing. Run: ollama pull qwen2.5:0.5b"; exit 1; }
docker image inspect cellbench-live >/dev/null 2>&1 || docker build -t cellbench-live .
docker rm -f cellbench-live >/dev/null 2>&1 || true
docker run --rm --name cellbench-live -p 8000:8000 --add-host=host.docker.internal:host-gateway cellbench-live
