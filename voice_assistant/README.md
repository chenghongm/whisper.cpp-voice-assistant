# Local Voice Assistant

A fully local, offline voice assistant pipeline for macOS.

```
Microphone → Whisper large-v3 (STT) → Ollama qwen2.5:7b (LLM) → Piper (TTS) → afplay
```

Supports **Chinese and English** with automatic language detection.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Python 3.10+
- [Ollama](https://ollama.com)
- Piper TTS voice model (downloaded separately)

---

## 1 — Python Setup

```bash
cd voice_assistant

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon note:** `faster-whisper` uses CPU int8 by default on arm64,
> which is fast and memory-efficient with no extra setup needed.

---

## 2 — Install & Start Ollama

```bash
# Install via the official installer
brew install ollama

# Or download from https://ollama.com/download

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

### Pull the default model

```bash
ollama pull qwen2.5:7b
```

You can use any other model by editing `OLLAMA_MODEL` in `main.py`.

---

## 3 — Piper TTS Setup

Piper needs a voice model file (`.onnx`) downloaded separately.

### Step 1 — Create the models directory

```bash
mkdir -p ~/piper_models
```

### Step 2 — Download a voice model

Choose a voice from the [Piper voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md).

**English (US) — recommended:**
```bash
cd ~/piper_models

# Download model + config
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz

# The .onnx file will be at ~/piper_models/en_US-lessac-medium.onnx
```

**Chinese (Mandarin) — alternative:**
```bash
cd ~/piper_models
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-zh-cn-huayan-medium.tar.gz
tar -xzf voice-zh-cn-huayan-medium.tar.gz
```

### Step 3 — Update the model path in `main.py`

```python
# main.py line ~35
PIPER_MODEL = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")
```

Change the filename to match whichever model you downloaded.

---

## 4 — Run the Assistant

Three processes. Start the first two once and leave them running:

```bash
ollama serve                  # terminal 1 — LLM
./start_whisper_server.sh     # terminal 2 — STT (see "Latency" below)

source .venv/bin/activate     # terminal 3
python main.py                # CLI, or: python app.py for the web UI
```

`start_whisper_server.sh` is optional — without it the assistant falls back to
in-process faster-whisper, which works but is ~3x slower per turn. On first run
it downloads the CoreML encoder (~1.2 GB) and the first transcription after that
compiles it (~30 s, once).

**Interaction:**

1. Recording starts automatically; just speak
2. Recording stops once you have actually finished — see "Endpointing" below
3. The assistant transcribes, queries the LLM, and speaks the response
4. Press **Ctrl+C** to quit

---

## 4b — Latency

Measured on an M5 with an 8-second utterance, time from end-of-speech to first
spoken word:

| Stage | Before | After | What changed |
|---|---|---|---|
| Waiting out your pause | 3.0 s | 1.8 s | see "Endpointing" — and the transcription now runs *inside* this wait rather than after it |
| Idle `sleep()` | 1.0 s | 0 s | removed from the loop |
| Transcription | 3.9 s | ~0 s | whisper.cpp server (CoreML + Metal) instead of faster-whisper on CPU int8 — CTranslate2 has no Metal backend. The 1.3 s it still costs is hidden inside the pause above, so it adds nothing |
| LLM first token | 10.7 s | 0.5 s | `think: false` — gemma4 is a reasoning model and was emitting ~1400 chars of hidden chain-of-thought before every reply |
| TTS first sentence | 0.1 s | 0.1 s | Piper is not a bottleneck (RTF 0.04) |
| **Total** | **~18.7 s** | **~2.4 s** | |

Transcription cost barely depends on how long you spoke, because Whisper pads
every input to a 30-second window. Measured on the server:

| Utterance | whisper.cpp (CoreML+Metal) | faster-whisper (CPU int8) |
|---|---|---|
| 1.3 s | 0.79 s | 3.8 s |
| 4.3 s | 0.97 s | 3.8 s |
| 7.8 s | 1.29 s | 3.9 s |
| 14.1 s | 1.67 s | 4.0 s |

So there is a ~0.8 s floor per turn no matter how short the answer, and long
answers are nearly free. Don't bother optimizing for terse replies.

### Endpointing — deciding when you have stopped talking

The first version used one number: stop after N seconds of quiet. That forces a
bad trade. Set it high (3 s) and every turn carries 3 s of dead air; set it low
(0.8 s) and it cuts you off the moment you pause to think.

There are three changes that remove the trade-off:

1. **Silero VAD instead of an RMS gate.** The old check was "is the microphone
   quieter than 300?", which counts trailing off quietly as silence. Silero is an
   actual speech classifier and costs ~2 ms per 100 ms tick, so it is free here.

2. **Transcribe during the pause, not after it.** At `EAGER_SILENCE` (0.6 s) a
   worker thread starts transcribing what you have said so far, while the main
   loop keeps listening. If you start talking again the result is discarded. If
   you do not, it is already finished by the time we commit — measured wait on
   Whisper at commit time: 0.00 s. This is what makes a long pause tolerance free.

3. **Check whether the sentence sounds finished.** That speculative transcript is
   free to inspect. If it ends without sentence-final punctuation, or trails off
   on a word like "and" / "because" / "the", you were probably mid-thought, so
   the wait extends to `EXTENDED_SILENCE` (4 s) instead of committing at 1.8 s.

Measured with synthesized speech (`spoke` = audio captured, `wait` = time blocked
on Whisper at the end):

| Scenario | Result |
|---|---|
| 1.2 s pause mid-sentence | kept listening, captured the whole sentence, wait 0.00 s |
| 3.0 s pause mid-sentence | "sounded unfinished" → extended, captured the whole sentence, wait 0.00 s |
| finished cleanly | committed 1.8 s after you stopped, wait 0.00 s |

If it still cuts you off, raise `COMMIT_SILENCE`. Because of (2) this costs you
latency one-for-one but nothing else — there is no longer any transcription time
hiding behind it. If it hangs on waiting for you, lower `EXTENDED_SILENCE`.

Two things to watch for when tuning further:

- **Don't set `num_ctx` in the Ollama options.** Changing it forces a model
  reload and costs ~8 s on the next turn.
- **Keep `keep_alive` generous** (`30m`). A cold gemma4 load is ~10 s, so an
  idle gap longer than the keep-alive window makes the next turn feel broken.

---

## 5 — Configuration

All options are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `MAX_RECORD_SECONDS` | `30` | Max recording length per turn |
| `EAGER_SILENCE` | `0.6` | Quiet before a speculative transcription starts |
| `COMMIT_SILENCE` | `1.8` | Quiet before your turn ends — raise if it cuts you off |
| `EXTENDED_SILENCE` | `4.0` | Used instead when the transcript sounds unfinished |
| `VAD_THRESHOLD` | `0.45` | Silero speech probability; lower = more sensitive |
| `SILENCE_THRESHOLD` | `300` | RMS fallback, only if Silero fails to load |
| `WHISPER_SERVER_URL` | `http://localhost:8178/inference` | whisper.cpp server; unreachable → local fallback |
| `WHISPER_MODEL_SIZE` | `distil-large-v3` | faster-whisper fallback model only |
| `WHISPER_LANGUAGE` | `"en"` | `None` = auto, or `"zh"` / `"en"` |
| `OLLAMA_MODEL` | `gemma4` | Any model pulled in Ollama |
| `OLLAMA_THINK` | `False` | Chain-of-thought off — the single biggest latency win |
| `OLLAMA_KEEP_ALIVE` | `"30m"` | How long Ollama holds the weights in memory |
| `OLLAMA_NUM_PREDICT` | `200` | Reply length cap; spoken answers should be short |
| `PIPER_MODEL` | `~/piper_models/en_US-lessac-medium.onnx` | Piper voice model path |

---

## 6 — Troubleshooting

**The assistant answers things you never said / seems to talk to itself:**

Almost always a dead microphone. When macOS has not granted microphone access,
`sounddevice` does not raise — it hands back a stream of zeros. Whisper, given
pure silence, hallucinates a stock filler word (usually `"you"`), the LLM
answers that in earnest, and the loop repeats. The transcript is the giveaway:

```
You: you
Assistant: Can you tell me what you'd like to discuss?
```

Fixes, in order:

1. **System Settings ▸ Privacy & Security ▸ Microphone** — allow your terminal
   (Terminal, iTerm, VS Code — whichever you launch from). The prompt only
   appears on first access and the first stream stays silent while it is pending,
   so the very first run can fail even after you click Allow.
2. **System Settings ▸ Sound ▸ Input** — confirm the selected device is the one
   you speak into. A paired Bluetooth speaker often grabs the default input slot.
3. Run `python main.py` and check the `Microphone:` line it prints at startup.

`main.py` now probes the input at startup and refuses to pretend: a silent stream
is reported rather than transcribed, a turn with no detected speech is dropped
instead of sent to the LLM, and Whisper's known silence fillers are discarded.

**No audio input / `sounddevice` error:**
```bash
# List available audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**Ollama not reachable:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
# If not, start it:
ollama serve
```

**TTS skipped (model not found):**
Check that `PIPER_MODEL` in `main.py` points to an existing `.onnx` file.

**Whisper downloading the model on first run:**
`faster-whisper` downloads `large-v3` (~3 GB) on first use and caches it in
`~/.cache/huggingface/hub`. Subsequent runs are instant.

---

## 7 — Future Improvements

See the TODO comments at the top of `main.py`:

- **Streaming transcription** — lower first-word latency
- **VAD (Voice Activity Detection)** — auto-stop recording when silence is detected
- **Interruptible TTS** — stop playback when the user starts speaking
- **Real-time streaming** — live ASR + streaming LLM output
- **Chat history / memory** — pass prior turns to Ollama for context
- **Wake word** — trigger recording with a keyword instead of Enter
