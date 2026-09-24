#!/usr/bin/env bash
# Start the whisper.cpp STT server used by the voice assistant.
#
# On Apple Silicon this runs the encoder through CoreML and the decoder through
# Metal: ~1.3 s for an 8 s utterance, versus ~3.9 s for in-process faster-whisper
# (CPU int8 — CTranslate2 has no Metal backend). The assistant falls back to
# faster-whisper automatically if this server is not running.
#
# Usage:  ./start_whisper_server.sh          (foreground)
#         ./start_whisper_server.sh &        (background)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/build/bin/whisper-server"
MODEL="$REPO/models/ggml-large-v3.bin"
PORT="${WHISPER_PORT:-8178}"

if [[ ! -x "$BIN" ]]; then
  echo "whisper-server not built. From $REPO run:"
  echo "  cmake -B build -DWHISPER_COREML=1 && cmake --build build -j --config Release"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Missing $MODEL — download it with:  $REPO/models/download-ggml-model.sh large-v3"
  exit 1
fi

# CoreML encoder: a second copy of Whisper's encoder, converted so it can run on
# the Apple Neural Engine instead of the GPU. This build has WHISPER_COREML=1 and
# WHISPER_COREML_ALLOW_FALLBACK=OFF, so whisper-server refuses to start without it.
COREML="$REPO/models/ggml-large-v3-encoder.mlmodelc"
if [[ ! -d "$COREML" ]]; then
  echo "Fetching CoreML encoder (~1.2 GB, one time)..."
  curl -L -o /tmp/ggml-large-v3-encoder.mlmodelc.zip \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-encoder.mlmodelc.zip"
  unzip -q -o /tmp/ggml-large-v3-encoder.mlmodelc.zip -d "$REPO/models/"
  echo "Note: the first transcription after this compiles the CoreML model (~30 s, once)."
fi

echo "whisper-server → http://localhost:$PORT/inference"
exec "$BIN" -m "$MODEL" -l en --port "$PORT" -bs 1 -t 8
