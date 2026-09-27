"""Optional microphone level and hybrid speech-to-text input.

Provides:
- Real-time audio stream capture with energy-based level metering.
- Automatic utterance segmentation (Voice Activity Detection).
- Tiered speech-to-text decoding via HybridSpeechTranscriber:
    1. Primary: Groq Cloud (whisper-large-v3-turbo)
    2. Fallback: Local faster-whisper (small) for flaky/absent network
    3. Fallback: Manual typed input if voice transcription is unavailable
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Optional

import numpy as np

try:
    from .speech_transcriber import (
        HybridSpeechTranscriber,
        TranscriptionResult,
        clean_transcribed_name,
    )
except (ImportError, ValueError):
    try:
        from src.engine.speech_transcriber import (
            HybridSpeechTranscriber,
            TranscriptionResult,
            clean_transcribed_name,
        )
    except (ImportError, ValueError):
        from speech_transcriber import (
            HybridSpeechTranscriber,
            TranscriptionResult,
            clean_transcribed_name,
        )

try:
    from .remote_audio_server import RemoteAudioReceiver
except (ImportError, ValueError):
    try:
        from src.engine.remote_audio_server import RemoteAudioReceiver
    except (ImportError, ValueError):
        from remote_audio_server import RemoteAudioReceiver

logger = logging.getLogger(__name__)


class OptionalAudioInput:
    """Bounded microphone input with tiered Groq / faster-whisper / manual transcription,

    supporting local microphone capture and remote laptop voice transmission.
    """

    sample_rate = 16_000

    def __init__(
        self,
        transcriber: Optional[HybridSpeechTranscriber] = None,
        model_path: str | Path | None = None,
        remote_port: Optional[int] = None,
    ) -> None:
        self._sounddevice = None
        self._stream = None
        self._decoder_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._audio_queue: Queue[bytes] = Queue(maxsize=32)
        self._text_queue: Queue[str] = Queue(maxsize=16)
        self._lock = threading.Lock()
        self._level = 0.0
        self._error: Optional[str] = None
        self._status_message: Optional[str] = None

        # Tiered STT orchestrator
        self.transcriber = transcriber or HybridSpeechTranscriber()

        # Remote receiver for companion laptop audio/name transmission
        port = remote_port or int(os.environ.get("LSFACE_REMOTE_VOICE_PORT", "5055"))
        self._remote_receiver = RemoteAudioReceiver(
            port=port,
            on_name_received=self._on_remote_name_received,
            transcriber=self.transcriber,
        )

        # Check audio hardware input availability
        try:
            import sounddevice

            self._sounddevice = sounddevice
            self.reason = f"Microphone level active (and remote port {self._remote_receiver.port})."
        except Exception:
            self._sounddevice = None
            self.reason = f"Local mic unavailable; listening for laptop audio on port {self._remote_receiver.port}."

        # Configure initial STT status message
        if self._sounddevice is not None and self.transcriber.has_any_stt:
            self.stt_reason = (
                f"Listening on local mic & remote laptop (port {self._remote_receiver.port}); or type below."
            )
        elif self._sounddevice is not None:
            self.stt_reason = (
                f"Local mic active, listening for laptop on port {self._remote_receiver.port}; or type below."
            )
        else:
            self.stt_reason = (
                f"Listening for laptop audio/name on port {self._remote_receiver.port}; or type below."
            )

    @property
    def available(self) -> bool:
        """Return True if local mic or remote laptop receiver is ready."""
        return self._sounddevice is not None or self._remote_receiver is not None

    @property
    def remote_port(self) -> int:
        return self._remote_receiver.port

    @property
    def stt_available(self) -> bool:
        """Return True if at least one speech-to-text provider is configured."""
        return self.transcriber.has_any_stt

    def start(self) -> bool:
        """Start local audio stream and remote companion receiver."""
        self._stop_event.clear()

        # Start remote companion receiver so laptop can stream audio or send transcribed names
        remote_ok = self._remote_receiver.start()

        mic_ok = False
        if self._sounddevice is not None:
            try:
                self._stream = self._sounddevice.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=1600,
                    callback=self._on_audio,
                )
                self._stream.start()
                mic_ok = True
            except Exception as exc:
                self.reason = (
                    f"Local mic unavailable; listening for laptop on port {self._remote_receiver.port}."
                )
                self._set_error(f"Local mic error ({exc}); listening on port {self._remote_receiver.port}.")

        if mic_ok and self.stt_available:
            self._decoder_thread = threading.Thread(
                target=self._decode_loop,
                name="hybrid-whisper-transcriber",
                daemon=True,
            )
            self._decoder_thread.start()

        return mic_ok or remote_ok

    def poll(self) -> tuple[float, list[str], str | None]:
        """Return the latest audio level, at most four transcripts, and any status/error message."""
        with self._lock:
            level = self._level
            status_or_error = self._status_message or self._error
            self._status_message = None
            self._error = None

        texts = []
        for _ in range(4):
            try:
                texts.append(self._text_queue.get_nowait())
            except Empty:
                break

        return level, texts, status_or_error

    def stop(self) -> None:
        """Stop audio stream, decoder thread, and remote companion receiver."""
        self._stop_event.set()
        if self._remote_receiver is not None:
            self._remote_receiver.stop()

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
            decoder.join(timeout=0.8)

        with self._lock:
            self._level = 0.0

    def _on_remote_name_received(self, name: str, provider: str = "laptop", model: str = "") -> None:
        """Callback invoked when a companion device (e.g. laptop) transmits a name."""
        cleaned = clean_transcribed_name(name)
        if not cleaned:
            return
        self._emit_text(cleaned)
        provider_label = f"{provider} ({model})".strip() if model else provider
        self._set_status(f"Received '{cleaned}' from laptop ({provider_label})")

    def push_audio_for_testing(self, pcm_bytes: bytes) -> None:
        """Push arbitrary 16-bit 16kHz PCM audio bytes into the decoder queue (for testing)."""
        blocksize_bytes = 1600 * 2  # 1600 samples * 2 bytes/sample = 3200 bytes
        for offset in range(0, len(pcm_bytes), blocksize_bytes):
            chunk = pcm_bytes[offset : offset + blocksize_bytes]
            try:
                self._audio_queue.put_nowait(chunk)
            except Full:
                pass

    def transcribe_pcm(self, pcm_bytes: bytes) -> TranscriptionResult:
        """Directly transcribe PCM bytes via the hybrid transcriber."""
        return self.transcriber.transcribe(pcm_bytes, sample_rate=self.sample_rate)

    def _on_audio(self, indata, _frames, _time_info, _status) -> None:
        """Callback from sounddevice InputStream."""
        samples = np.asarray(indata, dtype=np.float32) / 32768.0
        if samples.size:
            level = float(np.sqrt(np.mean(np.square(samples))))
            with self._lock:
                self._level = float(np.clip(level * 6.0, 0.0, 1.0))

        try:
            self._audio_queue.put_nowait(np.asarray(indata, dtype=np.int16).tobytes())
        except Full:
            # Drop audio block rather than allowing callback backlog
            pass

    def _decode_loop(self) -> None:
        """Continuous worker thread that segments speech utterances and transcribes them."""
        # VAD Parameters
        speech_threshold = 0.025
        silence_blocks_limit = 7  # 7 * 100ms = 700ms silence ends utterance
        min_speech_blocks = 3     # 3 * 100ms = 300ms minimum utterance
        max_speech_blocks = 50    # 5 * 1000ms = 5.0s maximum utterance length

        pre_speech_blocks: list[bytes] = []
        utterance_blocks: list[bytes] = []
        in_speech = False
        silence_blocks = 0

        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.15)
            except Empty:
                continue

            if not chunk:
                continue

            # Calculate RMS energy of 100ms chunk
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            is_speech_chunk = chunk_rms > speech_threshold

            if is_speech_chunk:
                if not in_speech:
                    in_speech = True
                    utterance_blocks = list(pre_speech_blocks)
                    utterance_blocks.append(chunk)
                    silence_blocks = 0
                else:
                    utterance_blocks.append(chunk)
                    silence_blocks = 0
            else:
                if in_speech:
                    utterance_blocks.append(chunk)
                    silence_blocks += 1
                    if silence_blocks >= silence_blocks_limit:
                        # Utterance completed
                        self._process_utterance(utterance_blocks, min_speech_blocks)
                        utterance_blocks = []
                        in_speech = False
                        silence_blocks = 0
                else:
                    # Maintain trailing 300ms pre-speech buffer
                    pre_speech_blocks.append(chunk)
                    if len(pre_speech_blocks) > 3:
                        pre_speech_blocks.pop(0)

            # Cap max utterance length
            if in_speech and len(utterance_blocks) >= max_speech_blocks:
                self._process_utterance(utterance_blocks, min_speech_blocks)
                utterance_blocks = []
                in_speech = False
                silence_blocks = 0

        # On shutdown, process any remaining speech
        if in_speech and len(utterance_blocks) >= min_speech_blocks:
            self._process_utterance(utterance_blocks, min_speech_blocks)

    def _process_utterance(self, blocks: list[bytes], min_blocks: int) -> None:
        """Transcribe an accumulated utterance block list."""
        if len(blocks) < min_blocks:
            return

        pcm_bytes = b"".join(blocks)
        try:
            result = self.transcriber.transcribe(pcm_bytes, sample_rate=self.sample_rate)
            if result.cleaned_name:
                self._emit_text(result.cleaned_name)
                if result.provider == "groq":
                    self._set_status(f"Transcribed '{result.cleaned_name}' (Groq whisper-v3-turbo)")
                elif result.is_fallback:
                    self._set_status(
                        f"Groq network error: transcribed '{result.cleaned_name}' (local faster-whisper fallback)"
                    )
                else:
                    self._set_status(
                        f"Transcribed '{result.cleaned_name}' (local faster-whisper small)"
                    )
            elif result.error:
                self._set_error(result.error)
        except Exception as exc:
            self._set_error(f"Speech transcription error: {exc}")

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

    def _set_status(self, message: str) -> None:
        with self._lock:
            self._status_message = message

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message


if __name__ == "__main__":
    audio = OptionalAudioInput()
    level, texts, err = audio.poll()
    assert 0.0 <= level <= 1.0
    audio.stop()
    print("Microphone Reason:", audio.reason)
    print("STT Reason:", audio.stt_reason)
    print("Transcriber Has Any STT:", audio.transcriber.has_any_stt)
