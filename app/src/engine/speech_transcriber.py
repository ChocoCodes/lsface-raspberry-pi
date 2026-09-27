"""Speech-to-text provider adapters for face enrollment name tagging.

Provides:
- GroqWhisperAdapter: Primary cloud provider using Groq whisper-large-v3-turbo.
- FasterWhisperAdapter: Local secondary provider using faster-whisper small model as fallback.
- HybridSpeechTranscriber: Orchestrator prioritizing Groq -> faster-whisper fallback -> manual typing.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional
import wave

import numpy as np

logger = logging.getLogger(__name__)


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Convert raw 16-bit PCM bytes to a valid in-memory WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def clean_transcribed_name(raw_text: str) -> str:
    """Normalize transcribed name, removing conversational prefixes and transcription punctuation."""
    text = " ".join(str(raw_text or "").split())
    # Remove conversational lead-in prefixes often spoken by users
    text = re.sub(
        r"^(?:my\s+name\s+is|i\s+am|i['’]m|this\s+is|it['’]s|its|hello\s+my\s+name\s+is|hi\s+my\s+name\s+is)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip leading/trailing punctuation common in Whisper outputs (. , ! ? " ' : ;)
    text = re.sub(r"^[\s.,!?;:\"']+", "", text)
    text = re.sub(r"[\s.,!?;:\"']+$", "", text)
    return text.strip()


@dataclass
class TranscriptionResult:
    text: str
    cleaned_name: str
    provider: str  # "groq", "faster-whisper", "manual"
    model: str
    latency_sec: float
    error: Optional[str] = None
    is_fallback: bool = False


class GroqWhisperAdapter:
    """Primary Cloud STT Adapter utilizing Groq Cloud Whisper API (whisper-large-v3-turbo)."""

    DEFAULT_MODEL = "whisper-large-v3-turbo"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        raw_model = model or os.environ.get("GROQ_WHISPER_MODEL", self.DEFAULT_MODEL)
        # Normalize colloquial name 'whisper-v3-turbo' to official Groq identifier 'whisper-large-v3-turbo'
        if raw_model == "whisper-v3-turbo":
            raw_model = "whisper-large-v3-turbo"
        self.model = raw_model
        self.timeout = timeout
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key:
            return
        try:
            import groq

            self._client = groq.Groq(api_key=self.api_key, timeout=self.timeout)
        except Exception as exc:
            logger.warning("Groq client initialization error: %s", exc)
            self._client = None

    @property
    def is_available(self) -> bool:
        """Return True if Groq package is installed and an API key is configured."""
        return bool(self.api_key and self._client is not None)

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16_000) -> TranscriptionResult:
        """Transcribe raw PCM bytes using Groq whisper-large-v3-turbo."""
        if not self.is_available:
            raise RuntimeError("Groq adapter is not available. Check GROQ_API_KEY environment variable.")

        t0 = time.perf_counter()
        wav_bytes = pcm16_to_wav(pcm_bytes, sample_rate=sample_rate)
        file_payload = ("speech.wav", wav_bytes, "audio/wav")

        response = self._client.audio.transcriptions.create(
            file=file_payload,
            model=self.model,
            prompt="Person's name for biometric identity enrollment.",
            temperature=0.0,
            timeout=self.timeout,
        )
        latency = time.perf_counter() - t0
        raw_text = getattr(response, "text", "") or ""
        cleaned = clean_transcribed_name(raw_text)

        return TranscriptionResult(
            text=raw_text,
            cleaned_name=cleaned,
            provider="groq",
            model=self.model,
            latency_sec=latency,
            is_fallback=False,
        )


class FasterWhisperAdapter:
    """Secondary Local STT Adapter utilizing faster-whisper with the 'small' model."""

    DEFAULT_MODEL = "small"

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name_or_path = (
            model_name_or_path or os.environ.get("FASTER_WHISPER_MODEL", self.DEFAULT_MODEL)
        )
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._available = False
        try:
            import faster_whisper

            self._faster_whisper = faster_whisper
            self._available = True
        except Exception as exc:
            logger.warning("faster-whisper is not available: %s", exc)
            self._faster_whisper = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """Return True if faster-whisper package is importable."""
        return self._available

    def _load_model(self):
        if self._model is None and self.is_available:
            with self._lock:
                if self._model is None:
                    self._model = self._faster_whisper.WhisperModel(
                        self.model_name_or_path,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
        return self._model

    def warm_up(self) -> None:
        """Preload the model in a background daemon thread so inference is instant on fallback."""
        if self.is_available and self._model is None:
            thread = threading.Thread(
                target=self._load_model,
                name="faster-whisper-warmup",
                daemon=True,
            )
            thread.start()

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16_000) -> TranscriptionResult:
        """Transcribe raw PCM bytes locally using faster-whisper small."""
        if not self.is_available:
            raise RuntimeError("faster-whisper is not installed or available.")

        t0 = time.perf_counter()
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return TranscriptionResult(
                text="",
                cleaned_name="",
                provider="faster-whisper",
                model=self.model_name_or_path,
                latency_sec=0.0,
            )

        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        model = self._load_model()
        with self._lock:
            segments, info = model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt="Name for biometric identity enrollment.",
            )
            raw_text = " ".join(seg.text.strip() for seg in segments).strip()

        latency = time.perf_counter() - t0
        cleaned = clean_transcribed_name(raw_text)

        return TranscriptionResult(
            text=raw_text,
            cleaned_name=cleaned,
            provider="faster-whisper",
            model=self.model_name_or_path,
            latency_sec=latency,
            is_fallback=False,
        )


class HybridSpeechTranscriber:
    """Tiered transcription orchestrator:

    1. Primary: Groq Cloud (whisper-large-v3-turbo).
    2. Fallback: Local faster-whisper (small model) if Groq fails or network is flaky/absent.
    3. Manual input fallback: if both STT options fail or are unavailable.
    """

    def __init__(
        self,
        groq_adapter: Optional[GroqWhisperAdapter] = None,
        faster_whisper_adapter: Optional[FasterWhisperAdapter] = None,
    ) -> None:
        self.groq = groq_adapter or GroqWhisperAdapter()
        self.faster_whisper = faster_whisper_adapter or FasterWhisperAdapter()
        self.last_provider_used: str = "none"
        self.last_error: Optional[str] = None
        self.fallback_occurred: bool = False

        # Asynchronously warm up the local model if available so fallback doesn't incur cold-start delay
        if self.faster_whisper.is_available:
            self.faster_whisper.warm_up()

    @property
    def has_any_stt(self) -> bool:
        """Return True if at least one speech-to-text provider is configured."""
        return self.groq.is_available or self.faster_whisper.is_available

    def get_initial_status(self) -> str:
        """Human-readable provider configuration status for the UI."""
        if self.groq.is_available:
            if self.faster_whisper.is_available:
                return "Microphone active (Primary: Groq whisper-v3-turbo, Fallback: faster-whisper small)."
            return "Microphone active (Primary: Groq whisper-v3-turbo; type your name if needed)."
        elif self.faster_whisper.is_available:
            return "Microphone active (Local: faster-whisper small; Groq API key not configured)."
        else:
            return "Speech-to-text unavailable; type your name below."

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16_000) -> TranscriptionResult:
        """Execute tiered transcription according to the configured hierarchy."""
        # Tier 1: Primary provider - Groq Cloud whisper-large-v3-turbo
        if self.groq.is_available:
            try:
                res = self.groq.transcribe(pcm_bytes, sample_rate=sample_rate)
                self.last_provider_used = "groq"
                self.fallback_occurred = False
                self.last_error = None
                return res
            except Exception as exc:
                self.fallback_occurred = True
                self.last_error = f"Groq primary failed ({exc}); fell back to local faster-whisper."
                logger.warning(self.last_error)

        # Tier 2: Secondary provider - Local faster-whisper small
        if self.faster_whisper.is_available:
            try:
                res = self.faster_whisper.transcribe(pcm_bytes, sample_rate=sample_rate)
                res.is_fallback = self.fallback_occurred or (not self.groq.is_available)
                self.last_provider_used = "faster-whisper"
                return res
            except Exception as exc:
                self.last_error = f"Local faster-whisper failed ({exc})."
                logger.error(self.last_error)

        # Tier 3: Tertiary fallback - Manual typing
        self.last_provider_used = "manual"
        return TranscriptionResult(
            text="",
            cleaned_name="",
            provider="manual",
            model="none",
            latency_sec=0.0,
            error=self.last_error or "All speech-to-text providers unavailable; type your name below.",
            is_fallback=True,
        )
