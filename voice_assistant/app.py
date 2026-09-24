#!/usr/bin/env python3
"""Voice Assistant — Gradio Web UI

Run:  python voice_assistant/app.py
Then open http://localhost:7860 in your browser.
"""

import os, sys, io, json, re, time, wave, platform, threading, queue
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
COMMIT_SILENCE     = 1.8   # stop listening after this much quiet — raise if cut off
EXTENDED_SILENCE   = 4.0   # ...but wait longer when the transcript sounds unfinished
MIN_SPEECH_SECONDS = 0.4   # your turn does not end until this much *speech* was heard
VAD_THRESHOLD      = 0.45  # Silero speech probability; lower = more sensitive
VAD_WINDOW_SECONDS = 1.5
SILENCE_THRESHOLD  = 300   # RMS fallback, used only if Silero fails to load
TTS_LENGTH_SCALE   = 1.2
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

# ── Shared mutable state (guarded by _lock where needed) ──────────────────────

_lock          = threading.Lock()
_system_prompt = DEFAULT_SYSTEM_PROMPT
_chat_history  = []
_status        = "Idle"
_stop_event    = threading.Event()
_is_running    = False
_whisper_mdl   = None
_piper_voice   = None
_tts_rate      = None

# ── Audio / AI helpers ─────────────────────────────────────────────────────────

def _trim_history():
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_chat_history) > max_msgs:
        del _chat_history[:len(_chat_history) - max_msgs]


def _append_transcript(user_text: str, assistant_text: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}]\nYou: {user_text}\n\nAssistant: {assistant_text}\n{'-'*60}\n\n")


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


def _listen_and_transcribe() -> tuple:
    """
    Record until you stop speaking, transcribing speculatively during the pause.

    Returns (text, seconds_recorded, seconds_spent_waiting_on_transcription).
    The third value is normally near zero: the transcription of everything up to
    EAGER_SILENCE runs on a worker thread while we are still confirming that the
    pause really was the end of your turn.
    """
    path = RECORDING_PATH
    vad          = _get_vad()
    chunk_frames = int(SAMPLE_RATE * 0.1)                      # 100 ms
    warmup_chunks   = 2                            # let the input stream settle
    eager_chunks    = int(EAGER_SILENCE / 0.1)
    commit_chunks   = int(COMMIT_SILENCE / 0.1)
    extended_chunks = int(EXTENDED_SILENCE / 0.1)
    min_speech   = int(MIN_SPEECH_SECONDS / 0.1)
    max_chunks   = int(MAX_RECORD_SECONDS / 0.1)
    window_samps = int(SAMPLE_RATE * VAD_WINDOW_SECONDS)

    global _status
    _status = "Listening — speak now"
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
                _wavfile.write(snap_path, SAMPLE_RATE, snap)
                text = _transcribe(snap_path)
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

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        for i in range(max_chunks):
            if _stop_event.is_set():
                break
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
                    _status = "Still listening…"
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
                        _status = "Waiting — that sounded unfinished…"
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
        _wavfile.write(path, SAMPLE_RATE, audio)   # worker timed out — do it now
        result["text"] = _transcribe(path)
    wait = time.perf_counter() - t_wait

    text = result.get("text", "")
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


def _ollama_tokens(user_text: str) -> Iterator[str]:
    _chat_history.append({"role": "user", "content": user_text})
    _trim_history()
    with _lock:
        system_prompt = _system_prompt
    messages      = [{"role": "system", "content": system_prompt}] + _chat_history
    payload       = {
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
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                full_response += token
                yield token
                if data.get("done"):
                    break
        if full_response.strip():
            _chat_history.append({"role": "assistant", "content": full_response.strip()})
    except Exception as exc:
        yield f"[Error: {exc}]"


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


def _tts_worker(q: queue.Queue):
    with sd.OutputStream(samplerate=_tts_rate, channels=1, dtype="int16") as stream:
        while True:
            sentence = q.get()
            if sentence is None:
                break
            try:
                clean = _clean_for_tts(sentence)
                if clean:
                    stream.write(_synthesize_pcm(clean))
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

def _run_loop():
    global _status, _is_running
    try:
        _status = "Loading models…"
        _load_models()

        while not _stop_event.is_set():
            _status = "Listening…"

            try:
                text, _spoke, _wait = _listen_and_transcribe()
                if _stop_event.is_set():
                    break
            except Exception as exc:
                _status = f"Listening failed: {exc}"
                continue

            if not text:
                _status = "Nothing heard — listening again"
                continue

            _status = f"You: {text[:70]}{'…' if len(text) > 70 else ''}"

            # LLM + TTS
            try:
                tts_buf = ""
                if _piper_voice:
                    tts_q  = queue.Queue()
                    player = threading.Thread(target=_tts_worker, args=(tts_q,), daemon=True)
                    player.start()
                    _status = "Thinking + Speaking…"

                    for token in _ollama_tokens(text):
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

                    if tts_buf.strip():
                        tts_q.put(tts_buf.strip())
                    tts_q.put(None)
                    player.join()
                else:
                    _status = "Thinking…"
                    for _ in _ollama_tokens(text):
                        pass

                if _chat_history and _chat_history[-1]["role"] == "assistant":
                    _append_transcript(text, _chat_history[-1]["content"])

            except Exception as exc:
                _status = f"LLM/TTS error: {exc}"
                continue

    except Exception as exc:
        _status = f"Fatal error: {exc}"
    finally:
        _is_running = False
        _status = "Stopped"

# ── Gradio UI callbacks ────────────────────────────────────────────────────────

def ui_toggle(btn_label: str):
    global _is_running
    if not _is_running:
        _stop_event.clear()
        _chat_history.clear()
        threading.Thread(target=_run_loop, daemon=True).start()
        _is_running = True
        return gr.update(value="Stop", variant="stop")
    else:
        _stop_event.set()
        _is_running = False
        return gr.update(value="Start Conversation", variant="primary")


def ui_save_prompt(new_prompt: str):
    global _system_prompt
    with _lock:
        _system_prompt = new_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    return "Saved"


def ui_get_status():
    return _status


def ui_get_transcript():
    if not os.path.exists(TRANSCRIPT_PATH):
        return "(no transcript yet)"
    with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
        return f.read()


def ui_export_transcript():
    return TRANSCRIPT_PATH if os.path.exists(TRANSCRIPT_PATH) else None


def ui_clear_conversation():
    """Clear in-memory chat history and the transcript file."""
    _chat_history.clear()
    if os.path.exists(TRANSCRIPT_PATH):
        os.remove(TRANSCRIPT_PATH)
    return "(no transcript yet)"

# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="Voice Assistant") as demo:
    gr.Markdown("# Voice Assistant")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### System Prompt")
            prompt_input = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                lines=8, label="", placeholder="Enter system prompt…",
            )
            with gr.Row():
                save_prompt_btn = gr.Button("Save Prompt", size="sm")
                prompt_msg      = gr.Markdown("")

            gr.Markdown("---")
            toggle_btn = gr.Button("Start Conversation", variant="primary", size="lg")
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
    toggle_btn.click(ui_toggle, inputs=[toggle_btn], outputs=[toggle_btn])
    clear_btn.click(ui_clear_conversation, outputs=[transcript_box])

    def _do_export():
        path = ui_export_transcript()
        return gr.update(value=path, visible=path is not None)

    export_btn.click(_do_export, outputs=[file_output])

    # Auto-refresh every 2 s
    timer = gr.Timer(2)
    timer.tick(ui_get_status,      outputs=[status_box])
    timer.tick(ui_get_transcript,  outputs=[transcript_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
