#!/usr/bin/env python3
"""Voice Assistant — Gradio Web UI

Run:  python voice_assistant/app.py
Then open http://localhost:7860 in your browser.
"""

import os, sys, io, json, re, time, wave, platform, threading, queue, tempfile
from typing import Iterator


def _ensure_venv():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python  = os.path.join(project_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        return
    if os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(project_dir, ".venv")):
        return
    os.execv(venv_python, [venv_python, os.path.abspath(__file__), *sys.argv[1:]])


_ensure_venv()

import requests
import numpy as np
import sounddevice as sd
from scipy.io import wavfile as _wavfile
from faster_whisper import WhisperModel
import gradio as gr

# ── Config ─────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16_000
# STT: whisper.cpp server (CoreML/Metal) first, in-process faster-whisper as fallback.
WHISPER_SERVER_URL     = "http://localhost:8178/inference"
WHISPER_SERVER_TIMEOUT = 30
TRANSCRIPTION_TIMEOUT = 90  # includes a possible local fallback
WHISPER_MODEL_SIZE = "distil-large-v3"   # fallback only (CPU int8, ~3x slower)
WHISPER_LANGUAGE   = "en"
OLLAMA_CHAT_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL       = "gemma4"
OLLAMA_THINK       = False   # gemma4 is a reasoning model; thinking costs ~13 s of TTFT
OLLAMA_KEEP_ALIVE  = "30m"   # keep weights resident; a cold reload costs ~10 s
OLLAMA_NUM_PREDICT = 200
MAX_HISTORY_TURNS  = 10
PIPER_MODEL        = os.path.expanduser("~/piper_models/en_US-lessac-medium.onnx")
RECORDING_PATH     = "/tmp/va_ui_recording.wav"
TRANSCRIPT_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.txt")
MAX_RECORD_SECONDS = 30
EAGER_SILENCE      = 0.6   # start speculative transcription after this much quiet
LIVE_ASR_INTERVAL  = 3.0   # preview long user turns using server-only snapshots
COMMIT_SILENCE     = 1.8   # stop listening after this much quiet — raise if cut off
EXTENDED_SILENCE   = 4.0   # ...but wait longer when the transcript sounds unfinished
MIN_SPEECH_SECONDS = 0.4   # your turn does not end until this much *speech* was heard
VAD_THRESHOLD      = 0.45  # Silero speech probability; lower = more sensitive
VAD_WINDOW_SECONDS = 1.5
AUDIO_STALL_SECONDS = 2.0  # no audio for this long = device went away; re-arm

# Echo rejection (speaker bleeding into the microphone)
ECHO_SETTLE_SECONDS = 0.6   # quiet gap after speaking before the mic opens
ECHO_MIN_RUN        = 4     # consecutive shared words that mark a sentence as echo
ECHO_OVERLAP        = 0.55  # ...and the fraction of the sentence they must cover
SILENCE_THRESHOLD  = 300   # RMS fallback, used only if Silero fails to load
TTS_LENGTH_SCALE   = 1.2
TTS_ABORT_SLICE    = 0.15  # seconds of audio per write — how fast Stop cuts speech
TTS_MAX_PLAYBACK   = 180   # hard ceiling on one reply's playback, in seconds
MAX_TTS_CHARS      = 80

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior technical interviewer at a Tier-1 tech company. "
    "Your tone is professional and slightly skeptical. "
    "Provide a 'Bridge Sentence' whenever I am stuck. "
    "When giving feedback, use a 'Scaffold' approach: tell me the 3 key concepts I "
    "should have used, then let me try again with those concepts. "
    "Keep responses concise and natural for speech. "
    "Avoid markdown, bullet points, or code blocks."
)

_SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])$')

# ── Shared mutable state ────────────────────────────────────────────────────────

_lock          = threading.Lock()
_system_prompt = DEFAULT_SYSTEM_PROMPT
_chat_history  = []
_status        = "Idle"
_session       = None
_loop_thread   = None
_pending_start = None  # None, fresh conversation (True), or resume (False)
_live_turn     = None  # visible, uncommitted transcript for the current session
_whisper_mdl   = None
_piper_voice   = None
_tts_rate      = None


class _Session:
    def __init__(self):
        self.stop = threading.Event()
        self.tts_abort = threading.Event()
        self.response_lock = threading.Lock()
        self.response = None
        self.player = None


def _set_status(session: _Session, value: str):
    global _status
    with _lock:
        if _session is session and not session.stop.is_set():
            _status = value

# ── Audio / AI helpers ─────────────────────────────────────────────────────────

def _trim_history():
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_chat_history) > max_msgs:
        del _chat_history[:len(_chat_history) - max_msgs]


def _set_live_turn(session: _Session, user_text: str, *, provisional=False):
    global _live_turn
    with _lock:
        if _session is session and not session.stop.is_set():
            previous = _live_turn if _live_turn and _live_turn["session"] is session else None
            _live_turn = {
                "session": session,
                "user": user_text,
                "assistant": previous["assistant"] if previous else "",
                "provisional": provisional,
            }


def _append_live_assistant(session: _Session, token: str):
    with _lock:
        if _live_turn and _live_turn["session"] is session and not session.stop.is_set():
            _live_turn["assistant"] += token


def _clear_live_turn(session: _Session):
    global _live_turn
    with _lock:
        if _live_turn and _live_turn["session"] is session:
            _live_turn = None


def _append_transcript(session: _Session, user_text: str, assistant_text: str):
    global _live_turn
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}]\nYou: {user_text}\n\nAssistant: {assistant_text}\n{'-'*60}\n\n")
        if _live_turn and _live_turn["session"] is session:
            _live_turn = None


def _get_whisper_device():
    if platform.machine() in ("arm64", "aarch64"):
        return "cpu", "int8"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


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


_HALLUCINATIONS = {
    "you", "you.", "thank you.", "thanks for watching!", "thank you very much.",
    "bye.", "okay.", "yeah.", "mm-hmm.", "[blank_audio]", "(silence)", ".", "!",
}


def _is_noise_transcript(text: str) -> bool:
    t = text.strip().lower()
    return t in _HALLUCINATIONS or len(re.findall(r"[A-Za-z\u4e00-\u9fff]", t)) < 2


def _listen_and_transcribe(session: _Session) -> tuple:
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
    preview_chunks = int(LIVE_ASR_INTERVAL / 0.1)
    window_samps = int(SAMPLE_RATE * VAD_WINDOW_SECONDS)

    _clear_live_turn(session)
    _set_status(session, "Listening — speak now")
    recorded, silence_count, speech_chunks = [], 0, 0
    pending, preview, speculated = None, None, False
    shown_samples = 0

    def _spawn(audio: np.ndarray, *, preview_only=False):
        """Transcribe one immutable snapshot; abandoned workers own no session state."""
        snap      = audio.copy()
        result = {"done": threading.Event(), "text": "", "error": None,
                  "samples": len(snap), "published": False}

        def work():
            snap_path = None
            try:
                fd, snap_path = tempfile.mkstemp(prefix="va_recording_", suffix=".wav")
                os.close(fd)
                _wavfile.write(snap_path, SAMPLE_RATE, snap)
                result["text"] = (_transcribe_via_server(snap_path) if preview_only
                                  else _transcribe(snap_path))
            except Exception as exc:
                result["error"] = exc
            finally:
                if snap_path is not None:
                    try:
                        os.remove(snap_path)
                    except OSError:
                        pass
                result["done"].set()

        threading.Thread(target=work, daemon=True).start()
        return result

    # Callback-driven capture rather than stream.read(), which blocks with no
    # timeout: a Bluetooth headset that drops its microphone profile when idle
    # simply stops delivering, and a blocking read parks there forever. That is
    # what made Stop look permanent — the flag below was never reached.
    audio_q: queue.Queue = queue.Queue()

    def _on_audio(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stall_ticks, stall_limit = 0, int(AUDIO_STALL_SECONDS / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk_frames, callback=_on_audio):
        i = -1
        while i < max_chunks - 1:
            if session.stop.is_set():
                break
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                stall_ticks += 1
                if stall_ticks >= stall_limit:
                    _set_status(session, "Microphone stopped sending audio — re-arming")
                    return "", 0.0, 0.0
                continue
            stall_ticks = 0
            i += 1
            recorded.append(chunk)

            if speech_chunks == 0 and i > 0 and i % 30 == 0:
                _set_status(session, f"Listening… ({i // 10}s, nothing heard yet)")

            tail   = np.concatenate(recorded[-int(VAD_WINDOW_SECONDS / 0.1):], axis=0).reshape(-1)
            tail   = (tail[-window_samps:] / 32768.0)
            speech = _is_speech(vad, tail)

            # A resumed sentence invalidates the pause snapshot, but a preview
            # of the continuous utterance is still useful while recording.
            if speech and speculated:
                pending, speculated = None, False
                _set_status(session, "Still listening…")
            for task in (preview, pending):
                if task and task["done"].is_set() and not task["published"]:
                    task["published"] = True
                    guess = task["text"]
                    if (not task["error"] and guess and
                            not _is_noise_transcript(guess) and
                            task["samples"] > shown_samples):
                        shown_samples = task["samples"]
                        _set_live_turn(session, guess, provisional=True)

            if i < warmup_chunks:
                continue

            if speech:
                speech_chunks += 1
                silence_count = 0
                if (speech_chunks >= min_speech and i > 0 and
                        i % preview_chunks == 0 and
                        (preview is None or preview["done"].is_set())):
                    preview = _spawn(np.concatenate(recorded, axis=0).reshape(-1),
                                     preview_only=True)
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
                guess = pending["text"] if pending and pending["done"].is_set() else ""
                if guess and _sounds_unfinished(guess) and silence_count < extended_chunks:
                    if silence_count == commit_chunks:
                        _set_status(session, "Waiting — that sounded unfinished…")
                    continue
                break

    # Stop pressed: drop everything captured and do not spend 1-2 s transcribing
    # audio nobody will hear a reply to.
    if session.stop.is_set() or not recorded:
        return "", 0.0, 0.0

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

    _set_status(session, "Transcribing…")
    t_wait = time.perf_counter()
    if pending is None:
        pending = _spawn(audio)
    while not pending["done"].wait(0.1):
        if session.stop.is_set():
            return "", duration, time.perf_counter() - t_wait
        if time.perf_counter() - t_wait >= TRANSCRIPTION_TIMEOUT:
            _set_status(session, "Transcription timed out — listening again")
            return "", duration, time.perf_counter() - t_wait
    if session.stop.is_set():
        return "", duration, time.perf_counter() - t_wait
    if pending["error"] is not None:
        raise pending["error"]
    wait = time.perf_counter() - t_wait
    text = pending["text"]
    if _is_noise_transcript(text):
        text = ""
    return text, duration, wait

# ─── Stage 2: Speech-to-Text via faster-whisper ──────────────────────────────

def _transcribe_via_server(path: str = RECORDING_PATH) -> str | None:
    """POST the WAV to whisper.cpp server; None if unreachable so we can fall back."""
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


def _transcribe(path: str = RECORDING_PATH) -> str:
    text = _transcribe_via_server(path)
    if text is not None:
        return text
    _load_fallback_whisper()
    segments, _ = _whisper_mdl.transcribe(
        path, language=WHISPER_LANGUAGE,
        beam_size=1, vad_filter=True, condition_on_previous_text=False,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ─── Echo rejection ───────────────────────────────────────────────────────────
#
# When the speaker and the microphone are the same device — a laptop without
# headphones, or a Bluetooth speaker like a soundcore — the microphone hears the
# assistant. Whisper faithfully transcribes it, the text arrives as if you had
# said it, and the model concludes you are parroting its own questions back.
#
# Bluetooth output lags 150-300 ms behind the write, so waiting for playback to
# "finish" is not enough; the tail of the last sentence is still in the air when
# the microphone opens. ECHO_SETTLE_SECONDS buys that time back, and the check
# below catches whatever still gets through.

def _echo_words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def _longest_run(a: list, b: list) -> int:
    """Longest run of consecutive words appearing in both lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def _strip_echo(text: str, last_reply: str) -> str:
    """
    Remove leading sentences the assistant just spoke. A *contiguous* run of
    shared words is the test, not a bag-of-words overlap: you will naturally
    reuse the interviewer's vocabulary, but you will not reproduce four words
    of it in a row by accident.
    """
    if not text or not last_reply:
        return text
    ref   = _echo_words(last_reply)
    parts = [p for p in re.split(r'(?<=[.!?。！？])\s+', text.strip()) if p.strip()]

    kept, dropping, dropped_any = [], True, False
    for part in parts:
        if dropping:
            words = _echo_words(part)
            run   = _longest_run(words, ref)
            is_echo = (
                len(words) >= 3 and run >= ECHO_MIN_RUN and run / len(words) >= ECHO_OVERLAP
            ) or (dropped_any and len(words) < 3)   # short tail right after an echo
            if is_echo:
                dropped_any = True
                continue
        dropping = False
        kept.append(part)

    out = " ".join(kept).strip()
    return "" if len(_echo_words(out)) < 2 else out


def _ollama_tokens(session: _Session, user_text: str) -> Iterator[str]:
    _chat_history.append({"role": "user", "content": user_text})
    _trim_history()
    with _lock:
        system_prompt = _system_prompt
    messages      = [{"role": "system", "content": system_prompt}] + list(_chat_history)
    payload       = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     True,
        "think":      OLLAMA_THINK,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options":    {"num_predict": OLLAMA_NUM_PREDICT},
    }
    tokens = queue.Queue()

    def fetch():
        resp = None
        try:
            # Requests cannot interrupt a blocked POST or read reliably from a
            # second thread. Keep it off the conversation/UI threads instead.
            with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True,
                               timeout=(5, 120)) as resp:
                with session.response_lock:
                    session.response = resp
                if session.stop.is_set():
                    return
                resp.raise_for_status()
                for line in resp.iter_lines(chunk_size=1):
                    if session.stop.is_set():
                        return
                    if line:
                        data = json.loads(line)
                        tokens.put(("token", data.get("message", {}).get("content", "")))
                        if data.get("done"):
                            break
        except Exception as exc:
            tokens.put(("error", exc))
        finally:
            with session.response_lock:
                if session.response is resp:
                    session.response = None
            tokens.put(("done", None))

    threading.Thread(target=fetch, daemon=True).start()
    full_response = ""
    while not session.stop.is_set():
        try:
            kind, value = tokens.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind == "token":
            full_response += value
            yield value
        elif kind == "error":
            yield f"[Error: {value}]"
        else:
            if full_response.strip() and not session.stop.is_set():
                _chat_history.append({"role": "assistant", "content": full_response.strip()})
            return


def _clean_for_tts(text: str) -> str:
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def _synthesize_pcm(text: str) -> np.ndarray:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        _piper_voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _tts_worker(session: _Session, q: queue.Queue):
    """
    Play queued sentences, staying responsive to Stop. A blocking q.get() and a
    whole-sentence stream.write() were the reason Stop appeared to do nothing:
    both hold on until the entire reply has been spoken. Here the queue is polled,
    playback is written in short slices, and abort() discards whatever is still
    buffered so the voice cuts off mid-word instead of finishing the sentence.
    """
    slice_frames = max(1, int(_tts_rate * TTS_ABORT_SLICE))
    with sd.OutputStream(samplerate=_tts_rate, channels=1, dtype="int16") as stream:
        while True:
            if session.stop.is_set() or session.tts_abort.is_set():
                stream.abort()
                return
            try:
                sentence = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if sentence is None:
                return
            try:
                clean = _clean_for_tts(sentence)
                if not clean:
                    continue
                audio = _synthesize_pcm(clean)
                for i in range(0, len(audio), slice_frames):
                    if session.stop.is_set() or session.tts_abort.is_set():
                        stream.abort()
                        return
                    stream.write(audio[i:i + slice_frames])
            except Exception as exc:
                print(f"[TTS error] {exc}")


def _load_fallback_whisper():
    """Load faster-whisper on demand — only when whisper-server is unreachable."""
    global _whisper_mdl
    if _whisper_mdl is None:
        device, compute = _get_whisper_device()
        _whisper_mdl = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute)


def _load_models():
    global _piper_voice, _tts_rate
    if _piper_voice is None and os.path.exists(PIPER_MODEL):
        try:
            from piper import PiperVoice
            _piper_voice = PiperVoice.load(PIPER_MODEL)
            _piper_voice.config.length_scale = TTS_LENGTH_SCALE
            _tts_rate = _piper_voice.config.sample_rate
        except Exception as exc:
            print(f"[Piper load error] {exc}")

# ── Conversation loop (runs in background thread) ──────────────────────────────

def _run_loop(session: _Session):
    try:
        _set_status(session, "Loading models…")
        _load_models()

        while not session.stop.is_set():
            _set_status(session, "Listening…")

            try:
                text, _spoke, _wait = _listen_and_transcribe(session)
                if session.stop.is_set():
                    break
            except Exception as exc:
                _set_status(session, f"Listening failed: {exc}")
                continue

            last_reply = (_chat_history[-1]["content"]
                          if _chat_history and _chat_history[-1]["role"] == "assistant" else "")
            cleaned = _strip_echo(text, last_reply)
            if cleaned != text:
                print(f"  [Echo removed] heard: {text!r}\n                kept: {cleaned!r}")
            text = cleaned

            if not text:
                _clear_live_turn(session)
                _set_status(session, "Nothing heard — listening again")
                continue

            _set_live_turn(session, text)
            _set_status(session, f"You: {text[:70]}{'…' if len(text) > 70 else ''}")

            # LLM + TTS
            try:
                tts_buf = ""
                if _piper_voice:
                    session.tts_abort.clear()
                    tts_q  = queue.Queue()
                    player = threading.Thread(target=_tts_worker, args=(session, tts_q), daemon=True)
                    session.player = player
                    player.start()
                    _set_status(session, "Thinking + Speaking…")

                    for token in _ollama_tokens(session, text):
                        if session.stop.is_set():
                            break
                        _append_live_assistant(session, token)
                        tts_buf += token
                        while True:
                            m = _SENTENCE_END.search(tts_buf)
                            if not m:
                                break
                            sentence = tts_buf[:m.start() + 1].strip()
                            tts_buf  = tts_buf[m.end():]
                            if sentence:
                                tts_q.put(sentence)
                        if len(tts_buf) >= MAX_TTS_CHARS:
                            for sep in (" ", "，", ",", "、"):
                                pos = tts_buf.rfind(sep, 0, MAX_TTS_CHARS)
                                if pos > MAX_TTS_CHARS // 3:
                                    tts_q.put(tts_buf[:pos].strip())
                                    tts_buf = tts_buf[pos + len(sep):]
                                    break
                            else:
                                tts_q.put(tts_buf.strip())
                                tts_buf = ""

                    if tts_buf.strip() and not session.stop.is_set():
                        tts_q.put(tts_buf.strip())
                    tts_q.put(None)
                    _set_status(session, "Speaking…")
                    deadline = time.monotonic() + TTS_MAX_PLAYBACK
                    while player.is_alive() and not session.stop.is_set():
                        player.join(timeout=0.1)
                        if time.monotonic() >= deadline:
                            _set_status(session, "Output device stalled — stopping playback")
                            session.tts_abort.set()
                            break
                    if not player.is_alive():
                        session.player = None
                    # Let the speaker actually go quiet before the mic reopens.
                    # Bluetooth output trails the write by 150-300 ms, so without
                    # this the tail of the last sentence lands in the next turn.
                    if not session.stop.is_set() and session.player is None:
                        _set_status(session, "…")
                        session.stop.wait(ECHO_SETTLE_SECONDS)
                else:
                    _set_status(session, "Thinking…")
                    for token in _ollama_tokens(session, text):
                        if session.stop.is_set():
                            break
                        _append_live_assistant(session, token)

                if session.stop.is_set():
                    if _chat_history and _chat_history[-1]["role"] == "user":
                        _chat_history.pop()      # interrupted before any reply
                    break

                # Do not reopen the microphone while the output device is still
                # owned by a stalled player. Stop remains responsive.
                while session.player is not None and session.player.is_alive():
                    _set_status(session, "Waiting for output device to release…")
                    session.player.join(timeout=0.1)
                    if session.stop.is_set():
                        break
                if session.stop.is_set():
                    break
                session.player = None

                if _chat_history and _chat_history[-1]["role"] == "assistant":
                    _append_transcript(session, text, _chat_history[-1]["content"])

            except Exception as exc:
                session.tts_abort.set()
                if session.player is not None:
                    while session.player.is_alive() and not session.stop.is_set():
                        session.player.join(timeout=0.1)
                    if not session.player.is_alive():
                        session.player = None
                _set_status(session, f"LLM/TTS error: {exc}")
                continue

    except Exception as exc:
        _set_status(session, f"Fatal error: {exc}")
    finally:
        if session.player is not None and session.player.is_alive():
            session.tts_abort.set()
            session.player.join()  # keep the mic reserved until playback exits
        _finish_session(session)

# ── Gradio UI callbacks ────────────────────────────────────────────────────────

def _launch_locked(clear_history: bool):
    """Start a session while holding _lock; only called after the old one exits."""
    global _session, _loop_thread, _status, _live_turn
    if clear_history:
        _chat_history.clear()
    _live_turn = None
    session = _Session()
    _session = session
    _status = "Starting…"
    _loop_thread = threading.Thread(target=_run_loop, args=(session,), daemon=True)
    _loop_thread.start()


def _finish_session(session: _Session):
    global _session, _status, _pending_start, _live_turn
    with _lock:
        if _session is not session:
            return
        if _live_turn and _live_turn["session"] is session:
            _live_turn = None
        _session = None
        requested = _pending_start
        _pending_start = None
        if requested is None:
            _status = "Stopped"
        else:
            _launch_locked(requested)


def _start_loop(clear_history: bool) -> bool:
    global _status, _pending_start
    with _lock:
        if _session is not None:
            if _session.stop.is_set():
                _pending_start = clear_history
                _status = "Finishing previous session; starting automatically…"
                return True
            _status = "Already running"
            return False
        _launch_locked(clear_history)
        return True


def _stop_loop() -> None:
    """Signal cancellation without blocking a Gradio callback on network I/O."""
    global _status
    with _lock:
        session = _session
        if session is None:
            return
        session.stop.set()
        session.tts_abort.set()
        _status = "Stopping…"

    def close_response():
        with session.response_lock:
            resp = session.response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    threading.Thread(target=close_response, daemon=True).start()


def _button_state():
    with _lock:
        session = _session
        queued = _pending_start is not None
    if queued:
        return gr.update(value="Starting…", variant="primary", interactive=False)
    if session is not None and session.stop.is_set():
        return gr.update(value="Stopping…", variant="stop", interactive=False)
    if session is not None:
        return gr.update(value="Stop", variant="stop", interactive=True)
    return gr.update(value="Start Conversation", variant="primary", interactive=True)


def ui_toggle():
    """Stop ends the session. Start begins a fresh one, discarding context."""
    with _lock:
        running = _session is not None and not _session.stop.is_set()
    if running:
        _stop_loop()
    else:
        _start_loop(clear_history=True)
    return _button_state()


def ui_resume():
    """Pick the conversation back up with the context it already has."""
    global _status
    with _lock:
        running = _session is not None and not _session.stop.is_set()
        has_history = bool(_chat_history)
    if running:
        with _lock:
            _status = "Already running"
    elif not has_history:
        with _lock:
            _status = "Nothing to resume — press Start Conversation"
    else:
        _start_loop(clear_history=False)
    return _button_state()


def ui_save_prompt(new_prompt: str):
    global _system_prompt
    with _lock:
        _system_prompt = new_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    return "Saved"


def ui_get_status():
    with _lock:
        return _status


def ui_get_transcript():
    with _lock:
        if os.path.exists(TRANSCRIPT_PATH):
            with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
                saved = f.read()
        else:
            saved = ""
        turn = dict(_live_turn) if _live_turn else None
    if turn:
        label = "You (recognizing…)" if turn["provisional"] else "You"
        live = f"{label}: {turn['user']}\n\nAssistant: {turn['assistant']}"
        return saved + live
    return saved or "(no transcript yet)"


def ui_export_transcript():
    return TRANSCRIPT_PATH if os.path.exists(TRANSCRIPT_PATH) else None


def ui_clear_conversation():
    """Clear in-memory chat history and the transcript file."""
    global _live_turn
    with _lock:
        _chat_history.clear()
        _live_turn = None
        if os.path.exists(TRANSCRIPT_PATH):
            os.remove(TRANSCRIPT_PATH)
    return "(no transcript yet)"

# ── Build UI ───────────────────────────────────────────────────────────────────

# Without max_lines a Textbox grows to fit its content, so the long default system
# prompt stretched the box instead of scrolling. Pinning max_lines to lines fixes
# the height; this CSS keeps the scrollbar visible and lets you drag it taller.
_CSS = """
.prompt-box textarea {
    overflow-y: auto !important;
    resize: vertical;
    scrollbar-gutter: stable;
}
"""

with gr.Blocks(title="Voice Assistant") as demo:
    gr.Markdown("# Voice Assistant")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### System Prompt")
            prompt_input = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                lines=8, max_lines=8, label="", placeholder="Enter system prompt…",
                elem_classes=["prompt-box"],
            )
            with gr.Row():
                save_prompt_btn = gr.Button("Save Prompt", size="sm")
                prompt_msg      = gr.Markdown("")

            gr.Markdown("---")
            toggle_btn = gr.Button("Start Conversation", variant="primary", size="lg")
            resume_btn = gr.Button("Resume", size="sm")
            status_box = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=2)

        with gr.Column(scale=2):
            gr.Markdown("### Transcript")
            transcript_box = gr.Textbox(
                value="(no transcript yet)",
                interactive=False, lines=26, max_lines=30, label="",
            )
            with gr.Row():
                clear_btn  = gr.Button("Clear Conversation", size="sm", variant="stop")
                export_btn = gr.Button("Export", size="sm")
            file_output = gr.File(label="Download", visible=False)

    # Events
    save_prompt_btn.click(ui_save_prompt, inputs=[prompt_input], outputs=[prompt_msg])
    toggle_btn.click(ui_toggle, outputs=[toggle_btn])
    resume_btn.click(ui_resume, outputs=[toggle_btn])
    clear_btn.click(ui_clear_conversation, outputs=[transcript_box])

    def _do_export():
        path = ui_export_transcript()
        return gr.update(value=path, visible=path is not None)

    export_btn.click(_do_export, outputs=[file_output])

    # Response tokens reach the transcript without waiting for speech playback.
    timer = gr.Timer(2)
    timer.tick(ui_get_status,      outputs=[status_box])
    timer.tick(_button_state,       outputs=[toggle_btn])
    transcript_timer = gr.Timer(0.5)
    transcript_timer.tick(ui_get_transcript, outputs=[transcript_box])


if __name__ == "__main__":
    # In Gradio 6 css/theme belong to launch(), not the Blocks constructor.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft(), css=_CSS)
