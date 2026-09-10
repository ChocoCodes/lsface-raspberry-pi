"""Optional microphone level and local speech-to-text input.

The app does not require either optional package.  Without them the caller can
keep a typed-name field active and continue safely.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading

import numpy as np


class OptionalAudioInput:
    """Bounded microphone input with an optional Vosk decoder."""

    sample_rate = 16_000

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._sounddevice = None
        self._vosk = None
        self._stream = None
        self._decoder_thread = None
        self._stop_event = threading.Event()
        self._audio_queue: Queue[bytes] = Queue(maxsize=8)
        self._text_queue: Queue[str] = Queue(maxsize=8)
        self._lock = threading.Lock()
        self._level = 0.0
        self._error = None

        try:
            import sounddevice
        except Exception:
            self.reason = "Optional microphone unavailable; type your name."
        else:
            self._sounddevice = sounddevice
            self.reason = "Microphone level active; type your name if needed."

        configured_model = model_path or os.environ.get("LSFACE_VOSK_MODEL")
        self._model_path = Path(configured_model) if configured_model else None
        if self._sounddevice is not None and self._model_path is not None and self._model_path.is_dir():
            try:
                import vosk
            except Exception:
                self.stt_reason = "Local speech-to-text unavailable; type your name."
            else:
                self._vosk = vosk
                self.stt_reason = "Listening for a name; you can edit the transcript."
        elif self._sounddevice is not None:
            self.stt_reason = "Speech-to-text model not configured; type your name."
        else:
            self.stt_reason = "Speech-to-text unavailable; type your name."

    @property
    def available(self) -> bool:
        return self._sounddevice is not None

    @property
    def stt_available(self) -> bool:
        return self._vosk is not None

    def start(self) -> bool:
        """Start the optional stream; return ``False`` when unavailable."""

        if not self.available:
            return False
        self._stop_event.clear()
        try:
            self._stream = self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=1600,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            self.reason = "Microphone unavailable; type your name."
            self._set_error(f"Microphone unavailable; type your name. ({exc})")
            self.stop()
            return False

        if self.stt_available:
            self._decoder_thread = threading.Thread(
                target=self._decode_loop,
                name="optional-name-transcriber",
                daemon=True,
            )
            self._decoder_thread.start()
        return True

    def poll(self) -> tuple[float, list[str], str | None]:
        """Return the latest level, at most four transcripts, and one error."""

        with self._lock:
            level = self._level
            error = self._error
            self._error = None

        texts = []
        for _ in range(4):
            try:
                texts.append(self._text_queue.get_nowait())
            except Empty:
                break
        return level, texts, error

    def stop(self) -> None:
        self._stop_event.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        decoder, self._decoder_thread = self._decoder_thread, None
        if decoder is not None and decoder is not threading.current_thread():
            decoder.join(timeout=0.5)
        with self._lock:
            self._level = 0.0

    def _on_audio(self, indata, _frames, _time_info, _status) -> None:
        samples = np.asarray(indata, dtype=np.float32) / 32768.0
        if samples.size:
            level = float(np.sqrt(np.mean(np.square(samples))))
            with self._lock:
                self._level = float(np.clip(level * 6.0, 0.0, 1.0))
        try:
            self._audio_queue.put_nowait(np.asarray(indata, dtype=np.int16).tobytes())
        except Full:
            # ponytail: drop an audio block rather than allowing callback backlog.
            pass

    def _decode_loop(self) -> None:
        try:
            model = self._vosk.Model(str(self._model_path))
            recognizer = self._vosk.KaldiRecognizer(model, self.sample_rate)
            while not self._stop_event.is_set():
                try:
                    audio = self._audio_queue.get(timeout=0.2)
                except Empty:
                    continue
                if recognizer.AcceptWaveform(audio):
                    payload = json.loads(recognizer.Result())
                    self._emit_text(payload.get("text", ""))
                else:
                    payload = json.loads(recognizer.PartialResult())
                    self._emit_text(payload.get("partial", ""))
        except Exception as exc:
            self._set_error(f"Speech-to-text stopped; type your name. ({exc})")

    def _emit_text(self, text: str) -> None:
        text = " ".join(str(text).split())
        if not text:
            return
        try:
            self._text_queue.put_nowait(text)
        except Full:
            try:
                self._text_queue.get_nowait()
            except Empty:
                pass
            try:
                self._text_queue.put_nowait(text)
            except Full:
                pass

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message


if __name__ == "__main__":
    audio = OptionalAudioInput()
    assert 0.0 <= audio.poll()[0] <= 1.0
    audio.stop()
    print(audio.reason)
