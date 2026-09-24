"""Cancellation tests that do not need audio devices or local model servers."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from voice_assistant import app


class SessionCancellationTests(unittest.TestCase):
    def setUp(self):
        with app._lock:
            self.assertIsNone(app._session)
            app._pending_start = None
            app._chat_history.clear()

    def tearDown(self):
        with app._lock:
            self.assertIsNone(app._session)
            app._pending_start = None
            app._chat_history.clear()

    def test_stop_does_not_wait_for_response_close(self):
        release = threading.Event()
        closing = threading.Event()

        class SlowResponse:
            def close(self):
                closing.set()
                release.wait(2)

        session = app._Session()
        session.response = SlowResponse()
        with app._lock:
            app._session = session
        try:
            start = time.monotonic()
            app._stop_loop()
            self.assertLess(time.monotonic() - start, 0.3)
            self.assertTrue(session.stop.is_set())
            self.assertTrue(closing.wait(1))
        finally:
            release.set()
            with app._lock:
                app._session = None

    def test_stop_while_ollama_has_not_returned_headers(self):
        posting = threading.Event()
        release = threading.Event()
        session = app._Session()

        def slow_post(*_args, **_kwargs):
            posting.set()
            release.wait(2)
            raise RuntimeError("request ended")

        with patch.object(app.requests, "post", side_effect=slow_post):
            consumer = threading.Thread(
                target=lambda: list(app._ollama_tokens(session, "hello")),
                daemon=True,
            )
            try:
                consumer.start()
                self.assertTrue(posting.wait(1))
                session.stop.set()
                consumer.join(timeout=0.3)
                self.assertFalse(consumer.is_alive())
                self.assertEqual(app._chat_history[-1]["role"], "user")
            finally:
                release.set()

    def test_completed_ollama_reply_keeps_history(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def raise_for_status(self):
                pass

            def iter_lines(self, *, chunk_size):
                if chunk_size != 1:
                    raise ValueError("expected promptly streamed lines")
                yield b'{"message":{"content":"Hello."},"done":false}'
                yield b'{"message":{"content":""},"done":true}'

        with patch.object(app.requests, "post", return_value=Response()):
            tokens = list(app._ollama_tokens(app._Session(), "hello"))
        self.assertEqual(tokens, ["Hello.", ""])
        self.assertEqual(app._chat_history[-1], {"role": "assistant", "content": "Hello."})

    def test_stop_during_transcription_returns_without_waiting_for_worker(self):
        transcribing = threading.Event()
        release = threading.Event()
        session = app._Session()
        chunks = [np.ones((1600, 1), dtype=np.int16) for _ in range(7)]
        chunks += [np.ones((1600, 1), dtype=np.int16) for _ in range(20)]

        class InputStream:
            def __init__(self, *, callback, **_kwargs):
                self.callback = callback

            def __enter__(self):
                for chunk in chunks:
                    self.callback(chunk, len(chunk), None, None)
                return self

            def __exit__(self, *_args):
                pass

        calls = iter([True] * 7 + [False] * 20)

        def slow_transcribe(_path):
            transcribing.set()
            release.wait(2)
            return "a real answer"

        result = []
        with app._lock:
            app._session = session
        try:
            with patch.object(app.sd, "InputStream", InputStream), \
                 patch.object(app, "_get_vad", return_value=None), \
                 patch.object(app, "_is_speech", side_effect=lambda *_: next(calls)), \
                 patch.object(app, "_transcribe", side_effect=slow_transcribe):
                worker = threading.Thread(
                    target=lambda: result.append(app._listen_and_transcribe(session)),
                    daemon=True,
                )
                worker.start()
                self.assertTrue(transcribing.wait(2))
                session.stop.set()
                worker.join(timeout=0.5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result[0][0], "")
        finally:
            release.set()
            with app._lock:
                app._session = None

    def test_resume_waits_for_old_session_without_clearing_its_stop(self):
        old_started = threading.Event()
        old_can_exit = threading.Event()
        new_started = threading.Event()
        sessions = []

        def fake_loop(session):
            sessions.append(session)
            if len(sessions) == 1:
                old_started.set()
                session.stop.wait(2)
                old_can_exit.wait(2)
            else:
                new_started.set()
                session.stop.wait(2)
            app._finish_session(session)

        with patch.object(app, "_run_loop", fake_loop):
            try:
                self.assertTrue(app._start_loop(clear_history=True))
                self.assertTrue(old_started.wait(1))
                app._chat_history.extend([
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ])
                app._stop_loop()
                self.assertTrue(app._start_loop(clear_history=False))
                self.assertFalse(new_started.is_set())
                self.assertTrue(sessions[0].stop.is_set())
                old_can_exit.set()
                self.assertTrue(new_started.wait(1))
                self.assertEqual(len(app._chat_history), 2)
            finally:
                old_can_exit.set()
                with app._lock:
                    active = app._session
                if active is not None:
                    active.stop.set()
                for _ in range(20):
                    with app._lock:
                        if app._session is None:
                            break
                    time.sleep(0.05)

    def test_stop_interrupts_playback_wait(self):
        player_started = threading.Event()
        session = app._Session()

        def fake_player(current_session, _queue):
            player_started.set()
            current_session.stop.wait(2)

        def fake_tokens(_session, _text):
            yield "Hello."

        with patch.object(app, "_load_models"), \
             patch.object(app, "_listen_and_transcribe", return_value=("hello", 1, 0)), \
             patch.object(app, "_ollama_tokens", side_effect=fake_tokens), \
             patch.object(app, "_tts_worker", side_effect=fake_player), \
             patch.object(app, "_piper_voice", object()):
            with app._lock:
                app._session = session
            loop = threading.Thread(target=app._run_loop, args=(session,), daemon=True)
            try:
                loop.start()
                self.assertTrue(player_started.wait(1))
                app._stop_loop()
                loop.join(timeout=0.5)
                self.assertFalse(loop.is_alive())
            finally:
                session.stop.set()
                loop.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
