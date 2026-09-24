#!/usr/bin/env python3
"""
Local Voice Assistant — Streaming Mode + Memory
Flow: Microphone (VAD) → Whisper large-v3 → Ollama chat (stream) → Piper (PCM) → sounddevice

# TODO (future upgrades):
#   - streaming transcription (faster first-token latency)
#   - interruptible TTS (detect mic energy, kill player thread mid-sentence)
#   - real-time streaming ASR + streaming LLM output simultaneously
#   - wake word support (e.g. openWakeWord) so no Enter key is needed
"""

import os
import sys
import io
import json
import re
import time
import wave
import platform
import threading
import queue
from typing import Optional, Iterator


def ensure_project_venv() -> None:
    """Re-run this script with the local project virtualenv when available."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir    = os.path.join(project_dir, ".venv")
    venv_python = os.path.join(project_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        return
    if os.path.realpath(sys.prefix) == os.path.realpath(venv_dir):
        return
    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


def require_dependency(module_name: str, package_name: Optional[str] = None):
    """Import a dependency or exit with an actionable environment hint."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError:
        package     = package_name or module_name
        project_dir = os.path.dirname(os.path.abspath(__file__))
        venv_python = os.path.join(project_dir, ".venv", "bin", "python")
        print(f"Missing Python package: {package}")
        print("This usually means main.py is running outside voice_assistant/.venv.")
        print(f"Install with: {venv_python} -m pip install -r {os.path.join(project_dir, 'requirements.txt')}")
        print(f"Run with:     {venv_python} {os.path.abspath(__file__)}")
        sys.exit(1)


ensure_project_venv()

requests     = require_dependency("requests")
np           = require_dependency("numpy")
sd           = require_dependency("sounddevice")
wavfile      = require_dependency("scipy.io.wavfile")
WhisperModel = require_dependency("faster_whisper").WhisperModel

# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS    = 1

# STT backend: whisper.cpp server (CoreML/Metal, ~3x faster on Apple Silicon) with
# automatic fallback to in-process faster-whisper when the server is not running.
# Start it with: ./start_whisper_server.sh   (see README)
WHISPER_SERVER_URL     = "http://localhost:8178/inference"
WHISPER_SERVER_TIMEOUT = 30      # seconds
WHISPER_MODEL_SIZE = "distil-large-v3"   # fallback model (CPU int8, ~3x slower)
# "zh" handles Chinese-primary + English code-switching and avoids hallucinations.
# Change to "en" for English-only, or None for auto-detect (not recommended: prone
# to the "Chinese character, Chinese character..." repetition loop on Chinese audio).
WHISPER_LANGUAGE   = "en"

# Ollama chat API (maintains multi-turn context server-side via messages list)
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "gemma4"
# gemma4 is a reasoning model: left on, it emits ~1400 chars of hidden chain-of-thought
# before the first spoken token (measured TTFT 13.5 s vs 0.2 s with it off).
OLLAMA_THINK      = False
OLLAMA_KEEP_ALIVE = "30m"   # keep weights resident; a cold reload costs ~10 s
OLLAMA_NUM_PREDICT = 200    # cap reply length — spoken answers should be short
SYSTEM_PROMPT   = (
    "You are a senior technical interviewer at a Tier-1 tech company. Your tone is professional and slightly skeptical. Provide a 'Bridge Sentence' whenever I am stuck. When giving feedback, use a 'Scaffold' approach: tell me the 3 key concepts I should have used, then let me try again with those concepts."
    "Keep responses concise and natural for speech. "
    "Avoid markdown, bullet points, or code blocks."
)
MAX_HISTORY_TURNS = 10  # sliding window: keep last N user+assistant pairs

PIPER_MODEL      = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")
RECORDING_PATH   = "/tmp/va_recording.wav"
TRANSCRIPT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.txt")

# ─── Endpointing (deciding when you have finished speaking) ───────────────────
#
# Speech/non-speech comes from Silero VAD (a real speech classifier, ~2 ms per
# call) rather than raw loudness, so trailing off quietly is not mistaken for
# silence the way an RMS gate does it.
#
# Two thresholds instead of one, so a generous pause tolerance costs ~nothing:
#   EAGER_SILENCE  — kick off transcription in the background, but do not commit.
#                    If you start speaking again, that result is thrown away.
#   COMMIT_SILENCE — actually stop listening. By now the background transcription
#                    is usually already finished, so the extra wait is free.
MAX_RECORD_SECONDS = 30
EAGER_SILENCE      = 0.6   # start speculative transcription after this much quiet
COMMIT_SILENCE     = 1.8   # stop listening after this much quiet — raise if cut off
EXTENDED_SILENCE   = 4.0   # ...but wait this long when the speculative transcript
                           #    looks like a sentence you have not finished yet
MIN_SPEECH_SECONDS = 0.4   # your turn does not end until this much *speech* was heard

VAD_THRESHOLD      = 0.45  # Silero speech probability; lower = more sensitive
VAD_WINDOW_SECONDS = 1.5   # rolling context fed to Silero each tick
SILENCE_THRESHOLD  = 300   # RMS fallback, used only if Silero fails to load

# TTS speed: 1.0 = normal, 1.2 = ~20% slower, 1.4 = noticeably slower
TTS_LENGTH_SCALE = 1.2

# TTS force-flush: don't wait forever for a sentence boundary in long clauses
MAX_TTS_CHARS = 80

_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])$')

# ─── Conversation history ─────────────────────────────────────────────────────

_chat_history: list = []  # [{role: "user"|"assistant", content: str}, ...]


def _trim_history() -> None:
    """Evict oldest turns so history never exceeds MAX_HISTORY_TURNS exchanges."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_chat_history) > max_msgs:
        del _chat_history[: len(_chat_history) - max_msgs]


def _append_transcript(user_text: str, assistant_text: str) -> None:
    """Append one exchange to the transcript file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"You: {user_text}\n\n")
        f.write(f"Assistant: {assistant_text}\n")
        f.write("-" * 60 + "\n\n")

# ─── Device Detection ─────────────────────────────────────────────────────────

def get_whisper_device():
    """Apple Silicon → CPU int8. CUDA available → CUDA float16. Fallback → CPU int8."""
    if platform.machine() in ("arm64", "aarch64"):
        print("  [Apple Silicon detected] Using CPU int8 compute.")
        return "cpu", "int8"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

def _microphone_is_live(seconds: float = 0.4) -> bool:
    """
    True if the default input actually delivers samples. macOS returns an all-zero
    stream rather than an error when microphone access is denied, so this is the
    only way to tell the difference before the first turn.
    """
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
            frames = [stream.read(int(SAMPLE_RATE * 0.1))[0].copy()
                      for _ in range(max(1, int(seconds / 0.1)))]
        return bool(np.concatenate(frames).any())
    except Exception as exc:
        print(f"  [Microphone check error] {exc}")
        return False


# ─── Stage 1+2: Listen with VAD endpointing, transcribing speculatively ──────

_vad_model = None


def _get_vad():
    """Load Silero VAD once. Returns None if unavailable (caller falls back to RMS)."""
    global _vad_model
    if _vad_model is None:
        try:
            from faster_whisper.vad import get_vad_model
            _vad_model = get_vad_model()
        except Exception as exc:
            print(f"  [Silero VAD unavailable, using RMS gate: {exc}]")
            _vad_model = False
    return _vad_model or None


def _is_speech(vad, tail: np.ndarray) -> bool:
    """
    True if the most recent audio looks like speech. Silero wants frames of 512
    samples, so the rolling window is trimmed to a multiple of that; the last few
    frames (~100 ms) decide, while the earlier ones only supply context.
    """
    if vad is None:
        rms = float(np.sqrt(np.mean(tail[-1600:].astype(np.float32) ** 2))) * 32768
        return rms >= SILENCE_THRESHOLD
    n = (len(tail) // 512) * 512
    if n < 512:
        return True
    probs = np.asarray(vad(tail[-n:].astype(np.float32)))
    return float(probs[-3:].max()) >= VAD_THRESHOLD


# Words that almost never end a finished thought. If the provisional transcript
# trails off on one of these, you were probably still mid-sentence when you paused.
_HANGING_WORDS = {
    "and", "or", "but", "so", "because", "if", "when", "while", "that", "which",
    "the", "a", "an", "to", "of", "for", "with", "in", "on", "at", "is", "are",
    "was", "were", "then", "like", "um", "uh", "er", "my", "i", "we", "it", "its",
    "this", "there", "would", "could", "should", "can", "will", "maybe", "about",
}


def _sounds_unfinished(text: str) -> bool:
    """
    Heuristic on the speculative transcript: does this look like a sentence that
    was cut off? Whisper punctuates, so a missing sentence-final mark is a decent
    signal, and a trailing conjunction or article is a strong one.
    """
    text = text.strip()
    if not text:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    if words and words[-1] in _HANGING_WORDS:
        return True
    return text[-1] not in ".!?。！？"


# What Whisper emits when handed near-silence. Even with the speech gate above,
# a door slam or a cough can get through, and these are never a real turn.
_HALLUCINATIONS = {
    "you", "you.", "thank you.", "thanks for watching!", "thank you very much.",
    "bye.", "okay.", "yeah.", "mm-hmm.", "[blank_audio]", "(silence)", ".", "!",
}


def _is_noise_transcript(text: str) -> bool:
    t = text.strip().lower()
    return t in _HALLUCINATIONS or len(re.findall(r"[A-Za-z\u4e00-\u9fff]", t)) < 2


def listen_and_transcribe(path: str) -> tuple:
    """
    Record until you stop speaking, transcribing speculatively during the pause.

    Returns (text, seconds_recorded, seconds_spent_waiting_on_transcription).
    The third value is normally near zero: the transcription of everything up to
    EAGER_SILENCE runs on a worker thread while we are still confirming that the
    pause really was the end of your turn.
    """
    vad          = _get_vad()
    chunk_frames = int(SAMPLE_RATE * 0.1)                      # 100 ms
    warmup_chunks   = 2                            # let the input stream settle
    eager_chunks    = int(EAGER_SILENCE / 0.1)
    commit_chunks   = int(COMMIT_SILENCE / 0.1)
    extended_chunks = int(EXTENDED_SILENCE / 0.1)
    min_speech   = int(MIN_SPEECH_SECONDS / 0.1)
    max_chunks   = int(MAX_RECORD_SECONDS / 0.1)
    window_samps = int(SAMPLE_RATE * VAD_WINDOW_SECONDS)

    print(f"  Listening (max {MAX_RECORD_SECONDS}s) — speak now...")
    recorded, silence_count, speech_chunks = [], 0, 0
    pending, speculated, generation = None, False, 0
    result = {}

    def _spawn(audio: np.ndarray):
        """
        Transcribe a snapshot on a worker thread; the result lands in `result`.
        Each attempt gets its own file and generation number, so a worker left
        over from a pause you spoke through cannot clobber the current one.
        """
        nonlocal generation
        generation += 1
        gen       = generation
        snap      = audio.copy()
        snap_path = f"{path}.{gen}.wav"

        def work():
            try:
                wavfile.write(snap_path, SAMPLE_RATE, snap)
                text = transcribe(snap_path)
                if gen == generation:          # still the live attempt
                    result["text"] = text
            finally:
                try:
                    os.remove(snap_path)
                except OSError:
                    pass

        t = threading.Thread(target=work, daemon=True)
        t.start()
        return t

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        for i in range(max_chunks):
            chunk, _ = stream.read(chunk_frames)
            recorded.append(chunk.copy())

            tail   = np.concatenate(recorded[-int(VAD_WINDOW_SECONDS / 0.1):], axis=0).reshape(-1)
            tail   = (tail[-window_samps:] / 32768.0)
            speech = _is_speech(vad, tail)

            if i < warmup_chunks:
                continue

            if speech:
                speech_chunks += 1
                if speculated:
                    print("  …still listening", flush=True)   # you resumed; discard the guess
                silence_count, pending, speculated = 0, None, False
                result.clear()
                continue

            # Nothing said yet — this is not a pause, you simply have not started.
            # Without this gate the turn would end on pure silence a second or two
            # after the assistant stopped talking, and Whisper hallucinates a word
            # like "you" out of the noise, which the LLM then earnestly answers.
            if speech_chunks < min_speech:
                continue

            silence_count += 1

            # Pause long enough to be worth guessing on, but not to commit yet.
            if silence_count == eager_chunks:
                pending    = _spawn(np.concatenate(recorded, axis=0).reshape(-1))
                speculated = True

            if silence_count >= commit_chunks:
                # The speculative transcript is usually ready by now (it started
                # 1.2 s ago). If it reads as unfinished, give you longer to resume.
                guess = result.get("text", "")
                if guess and _sounds_unfinished(guess) and silence_count < extended_chunks:
                    if silence_count == commit_chunks:
                        print(f"  …waiting, that sounded unfinished ({guess[-40:]!r})", flush=True)
                    continue
                break

    audio    = np.concatenate(recorded, axis=0).reshape(-1)
    duration = len(audio) / SAMPLE_RATE

    if not audio.any():
        # macOS hands out an all-zero stream instead of an error when microphone
        # access has not been granted. Left unreported, Whisper hallucinates a
        # word like "you" out of the silence and the LLM answers it, which looks
        # like the assistant talking to itself.
        print("  [Microphone delivered pure silence — no audio at all.]")
        print("  [Check System Settings ▸ Privacy & Security ▸ Microphone, and that")
        print("   the right input device is selected in Sound settings.]")
        return "", duration, 0.0

    if speech_chunks < min_speech:      # timed out without hearing you at all
        return "", duration, 0.0

    t_wait = time.perf_counter()
    if pending is not None:
        pending.join(timeout=WHISPER_SERVER_TIMEOUT)
    if "text" not in result:      # hit MAX_RECORD_SECONDS mid-sentence, or the
        wavfile.write(path, SAMPLE_RATE, audio)   # worker timed out — do it now
        result["text"] = transcribe(path)
    wait = time.perf_counter() - t_wait

    text = result.get("text", "")
    if _is_noise_transcript(text):
        print(f"  [Discarded as noise: {text!r}]")
        text = ""

    print(f"  Stopped after {duration:.1f}s.")
    return text, duration, wait

# ─── Stage 2: Speech-to-Text via faster-whisper ──────────────────────────────

def _make_silence_probe() -> str:
    """A 0.2 s silent WAV used once at startup to detect whether the server is up."""
    path = "/tmp/va_probe.wav"
    if not os.path.exists(path):
        wavfile.write(path, SAMPLE_RATE, np.zeros(SAMPLE_RATE // 5, dtype=np.int16))
    return path


_SILENCE_PROBE = _make_silence_probe()


def _transcribe_via_server(path: str) -> Optional[str]:
    """
    POST the WAV to a running whisper.cpp server. Returns text, or None if the
    server is unreachable so the caller can fall back to faster-whisper.
    """
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                WHISPER_SERVER_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"temperature": "0", "response_format": "json"},
                timeout=WHISPER_SERVER_TIMEOUT,
            )
        resp.raise_for_status()
        return " ".join(resp.json().get("text", "").split()).strip()
    except requests.exceptions.RequestException:
        return None


_fallback_model = None


def _transcribe_local(path: str) -> str:
    """In-process faster-whisper. Model is loaded lazily, only if ever needed."""
    global _fallback_model
    if _fallback_model is None:
        device, compute_type = get_whisper_device()
        print(f"  [Loading fallback Whisper '{WHISPER_MODEL_SIZE}' {device}/{compute_type}...]")
        _fallback_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    segments, _ = _fallback_model.transcribe(
        path,
        language=WHISPER_LANGUAGE,
        beam_size=1,                       # beam 5 costs ~0.3 s for no measurable gain here
        vad_filter=True,
        condition_on_previous_text=False,  # breaks "Chinese character..." repetition loops
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe(path: str) -> str:
    """Return the recognized text from a WAV file (server first, local fallback)."""
    text = _transcribe_via_server(path)
    if text is not None:
        return text
    print("  [whisper-server unreachable — using slower in-process fallback]")
    return _transcribe_local(path)

# ─── Stage 3: Stream tokens from Ollama (chat API + history) ─────────────────

def _ollama_tokens(user_text: str) -> Iterator[str]:
    """
    Append user_text to history, stream raw tokens from /api/chat, yield each token.
    Appends the completed reply to history when the stream ends.
    Sentence-buffering for TTS is handled by the caller (main loop), so this
    function stays focused on network I/O and history management.
    """
    _chat_history.append({"role": "user", "content": user_text})
    _trim_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _chat_history
    payload  = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     True,
        "think":      OLLAMA_THINK,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options":    {"num_predict": OLLAMA_NUM_PREDICT},
    }

    full_response = ""
    try:
        with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data   = json.loads(line)
                token  = data.get("message", {}).get("content", "")
                full_response += token
                yield token
                if data.get("done"):
                    break

        if full_response.strip():
            _chat_history.append({"role": "assistant", "content": full_response.strip()})

    except requests.exceptions.ConnectionError:
        yield "[Error: Cannot reach Ollama — is it running? Try: ollama serve]"
    except requests.exceptions.Timeout:
        yield "[Error: Ollama timed out after 120 s.]"
    except Exception as exc:
        yield f"[Error: {exc}]"

# ─── Stage 4+5: Piper TTS → sounddevice ──────────────────────────────────────

def load_piper_voice():
    """Load Piper voice model. Returns (voice, sample_rate) or (None, None)."""
    if not os.path.exists(PIPER_MODEL):
        print(f"  [TTS skipped] Model not found: {PIPER_MODEL}")
        print("  See README for Piper model download instructions.")
        return None, None
    try:
        from piper import PiperVoice  # type: ignore
        voice = PiperVoice.load(PIPER_MODEL)
        voice.config.length_scale = TTS_LENGTH_SCALE
        return voice, voice.config.sample_rate
    except ImportError:
        print("  [TTS skipped] piper-tts not installed (pip install piper-tts).")
        return None, None
    except Exception as exc:
        print(f"  [TTS load error] {exc}")
        return None, None


def _synthesize_to_pcm(voice, text: str):
    """Synthesize one sentence to int16 PCM via synthesize_wav → BytesIO → ndarray."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    buf.seek(0)
    with wave.open(buf, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _clean_for_tts(text: str) -> str:
    """Strip markdown symbols so Piper doesn't read them aloud."""
    text = re.sub(r'```[\s\S]*?```', '', text)          # fenced code blocks
    text = re.sub(r'`([^`]*)`', r'\1', text)            # inline code
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text) # bold/italic
    text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)   # underscore bold/italic
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)      # blockquotes
    return text.strip()


def _tts_player_worker(voice, sample_rate: int, q: queue.Queue) -> None:
    """
    Background thread: pull sentences, synthesize PCM, write to one persistent
    sounddevice OutputStream. One open stream avoids inter-sentence gaps.
    Exits on None poison pill; OutputStream.__exit__ drains the buffer.
    """
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        while True:
            sentence = q.get()
            if sentence is None:
                break
            try:
                clean = _clean_for_tts(sentence)
                if clean:
                    audio = _synthesize_to_pcm(voice, clean)
                    stream.write(audio)
            except Exception as exc:
                print(f"\n  [TTS error] {exc}")

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 54)
    print("  Local Voice Assistant  (Streaming + Memory)")
    print(f"  STT : whisper.cpp server (fallback: {WHISPER_MODEL_SIZE})")
    print(f"  LLM : {OLLAMA_MODEL}  (chat, history={MAX_HISTORY_TURNS} turns)")
    print("  TTS : Piper → sounddevice")
    print(f"  VAD : Silero endpointing ({COMMIT_SILENCE}s pause tolerance)")
    print("=" * 54)
    print("Press Ctrl+C at any time to quit.\n")

    if _transcribe_via_server(_SILENCE_PROBE) is not None:
        print(f"Whisper: whisper.cpp server at {WHISPER_SERVER_URL}")
    else:
        print(f"Whisper: server not up — will fall back to in-process {WHISPER_MODEL_SIZE}.")
        print("         For ~5x faster transcription run: ./start_whisper_server.sh")

    if not _microphone_is_live():
        print("Microphone check FAILED: the input stream is delivering pure silence.")
        print("  • System Settings ▸ Privacy & Security ▸ Microphone — allow your terminal")
        print("  • System Settings ▸ Sound ▸ Input — pick the device you actually speak into")
        print("  Continuing anyway, but nothing you say will be heard.\n")
    else:
        print(f"Microphone: {sd.query_devices(sd.default.device[0])['name']}")

    voice, tts_sample_rate = load_piper_voice()
    if voice:
        print(f"Piper voice ready ({tts_sample_rate} Hz).")
    print()

    while True:
        # ── Auto-start after a brief pause ────────────────────────────────
        try:
            print("Listening … (Ctrl+C to quit)")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

        # ── Stage 1+2: Listen, endpoint, transcribe (overlapped) ──────────
        t0 = time.perf_counter()
        try:
            text, spoke, wait = listen_and_transcribe(RECORDING_PATH)
        except Exception as exc:
            print(f"  [Listening error] {exc}\n")
            continue
        t2 = time.perf_counter()
        print(f"  [spoke {spoke:.1f}s | endpoint+transcribe {t2-t0-spoke:.1f}s, "
              f"of which {wait:.1f}s was actually waiting on Whisper]")

        if not text:
            print("  [No speech heard this turn — listening again]\n")
            continue
        print(f"\nYou: {text}\n")

        # ── Stage 3+4+5: Stream LLM → TTS → play (pipelined) ─────────────
        # Print every token immediately for visible streaming, while batching
        # complete sentences into the TTS queue for natural speech prosody.
        print("Assistant: ", end="", flush=True)
        t3 = time.perf_counter()

        tts_buf = ""
        if voice:
            tts_queue = queue.Queue()
            player = threading.Thread(
                target=_tts_player_worker,
                args=(voice, tts_sample_rate, tts_queue),
                daemon=True,
            )
            player.start()

            for token in _ollama_tokens(text):
                print(token, end="", flush=True)   # visible token-by-token streaming
                tts_buf += token

                # Flush complete sentences to TTS
                while True:
                    m = _SENTENCE_END.search(tts_buf)
                    if not m:
                        break
                    sentence = tts_buf[: m.start() + 1].strip()
                    tts_buf  = tts_buf[m.end():]
                    if sentence:
                        tts_queue.put(sentence)

                # Force-flush long clauses (handles Chinese without sentence-final punctuation)
                if len(tts_buf) >= MAX_TTS_CHARS:
                    for sep in (" ", "，", ",", "、"):
                        pos = tts_buf.rfind(sep, 0, MAX_TTS_CHARS)
                        if pos > MAX_TTS_CHARS // 3:
                            tts_queue.put(tts_buf[:pos].strip())
                            tts_buf = tts_buf[pos + len(sep):]
                            break
                    else:
                        tts_queue.put(tts_buf.strip())
                        tts_buf = ""

            if tts_buf.strip():
                tts_queue.put(tts_buf.strip())
            tts_queue.put(None)
            player.join()
        else:
            for token in _ollama_tokens(text):
                print(token, end="", flush=True)

        t4 = time.perf_counter()

        if _chat_history and _chat_history[-1]["role"] == "assistant":
            _append_transcript(text, _chat_history[-1]["content"])
        turns = len(_chat_history) // 2
        print(f"\n  [LLM+TTS {t4-t3:.1f}s | history: {turns} turn(s)]\n")


if __name__ == "__main__":
    main()
