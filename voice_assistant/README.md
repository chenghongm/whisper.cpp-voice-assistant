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

## 4c — Web UI controls (`app.py`)

| Button | Effect |
|---|---|
| **Start Conversation** | Begins a new session and **clears** the chat history |
| **Stop** | Ends the session. Cuts off mid-word if it is speaking |
| **Resume** | Restarts the loop **keeping** the context it already had |
| **Clear Conversation** | Wipes the in-memory history and the transcript file |

The transcript box refreshes every 0.5 s. A completed user utterance appears as
soon as recognition finishes, and the assistant's text appears as Ollama tokens
arrive, while speech is still playing. During long utterances, the whisper.cpp
server can provide a provisional snapshot about every 3 s; it is labeled
"recognizing" and may change when the turn ends. Without the server, the local
fallback still recognizes the completed turn, but has no during-speech preview.
Only completed exchanges are saved to `conversation.txt` for export.

Stop signals the current session immediately. Start or Resume requested while it
is stopping begins automatically after the old session releases its microphone
and speaker. A stalled audio device can still delay that release.

The blocking stages are handled as follows:

- **Audio capture.** `stream.read()` blocks with no timeout. A Bluetooth mic that
  drops its profile when idle just stops delivering, so the loop parked there
  forever — Stop never took effect and Start reported the old session as still
  stopping. Capture is now callback-driven with a polled queue, and a device that
  goes quiet for `AUDIO_STALL_SECONDS` is reported and re-armed instead of hanging.

- **TTS playback.** `queue.get()` and a whole-sentence `stream.write()` both block
  until the entire reply has been spoken. The queue is now polled, audio is written
  in 0.15 s slices, and `abort()` discards the buffer so the voice cuts mid-word.
- **The Ollama stream.** The request and response reader run on a worker thread.
  The conversation checks for Stop every 0.1 s, including before response headers
  arrive. Closing the response is also done off the UI thread.
- **Transcription.** Speculative and final transcription run on worker threads.
  Stop interrupts the conversation's wait for a result, and the status says
  "Transcribing…" while it waits.
- **Restarting.** Each session has its own cancellation signal and network
  response. Start or Resume waits for the previous conversation and playback to
  exit, then starts automatically; old workers cannot alter the new session's
  history or status.

**If it accuses you of repeating its own questions:**

The microphone is hearing the assistant. This happens whenever the speaker and the
microphone are the same device — a laptop without headphones, or a Bluetooth
speaker. Whisper transcribes the assistant's own voice, it arrives as if you had
said it, and the model reasonably concludes you are parroting it back. The
transcript gives it away: a `You:` line that repeats the previous `Assistant:`
line, often just its tail, because Bluetooth output trails the write by 150-300 ms
and the last sentence is still in the air when the microphone opens.

The first thing to rule out is the loop talking over itself: if the UI says
"Listening…" while you can still hear the assistant, the microphone is open during
playback and every reply will echo. Waiting for playback must not be capped — one
version of this loop used `player.join(timeout=10.0)`, and since most replies take
longer than 10 s to speak, the loop moved on and recorded the rest of the reply
every single turn. It now waits for playback to genuinely finish; `TTS_MAX_PLAYBACK`
is only a ceiling for a wedged output device, and it cuts playback through a
separate signal rather than ending the session.

Two further defences, both on by default:

- `ECHO_SETTLE_SECONDS` (0.6 s) keeps the microphone shut until the speaker is
  really quiet. An earlier version of this loop had a `time.sleep(1.0)` at the top
  that was doing this by accident; removing it as dead latency brought the problem
  out into the open.
- `_strip_echo()` drops leading sentences the assistant just spoke. The test is a
  *contiguous* run of shared words (`ECHO_MIN_RUN`, 4) covering enough of the
  sentence (`ECHO_OVERLAP`, 0.55) — you will reuse the interviewer's vocabulary
  constantly, but you will not reproduce four of its words in a row by accident.
  When it fires, the console prints what it heard and what it kept.

**The real fix is headphones.** Echo cancellation at the text level is a backstop,
not a substitute: anything the assistant says that you then genuinely repeat will
be stripped too, and a long enough echo can still slip through mid-sentence.

**If it says the microphone stopped sending audio:** that is a real dropout, not a
bug — Bluetooth headsets commonly release the mic when idle. The loop re-arms by
itself. If it keeps happening, switch Sound ▸ Input to the built-in microphone.

## 5 — Configuration

All options are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `MAX_RECORD_SECONDS` | `30` | Max recording length per turn |
| `EAGER_SILENCE` | `0.6` | Quiet before a speculative transcription starts |
| `COMMIT_SILENCE` | `1.8` | Quiet before your turn ends — raise if it cuts you off |
| `EXTENDED_SILENCE` | `4.0` | Used instead when the transcript sounds unfinished |
| `ECHO_SETTLE_SECONDS` | `0.6` | Quiet gap after speaking before the mic reopens |
| `ECHO_MIN_RUN` | `4` | Consecutive shared words that mark a sentence as echo |
| `ECHO_OVERLAP` | `0.55` | Fraction of the sentence that run must cover |
| `AUDIO_STALL_SECONDS` | `2.0` | Silence from the *device* before declaring a dropout |
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
