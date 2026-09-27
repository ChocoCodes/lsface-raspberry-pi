"""Unit and integration tests for Whisper Groq adapter, faster-whisper fallback, and hybrid transcription."""
from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

from app.src.engine.speech_transcriber import (
    FasterWhisperAdapter,
    GroqWhisperAdapter,
    HybridSpeechTranscriber,
    TranscriptionResult,
    clean_transcribed_name,
    pcm16_to_wav,
)
from app.src.engine.audio_input import OptionalAudioInput
from app.src.views.voice_recognition import VoiceRecognitionScreen


class SpeechTranscriberCleaningTests(unittest.TestCase):
    def test_clean_conversational_prefixes(self):
        self.assertEqual(clean_transcribed_name("My name is John Doe"), "John Doe")
        self.assertEqual(clean_transcribed_name("my name is Alice"), "Alice")
        self.assertEqual(clean_transcribed_name("I am Robert Smith"), "Robert Smith")
        self.assertEqual(clean_transcribed_name("I'm Bob"), "Bob")
        self.assertEqual(clean_transcribed_name("This is Carol"), "Carol")
        self.assertEqual(clean_transcribed_name("Hello my name is Daniel"), "Daniel")
        self.assertEqual(clean_transcribed_name("It's Evan"), "Evan")

    def test_clean_whisper_punctuation_artifacts(self):
        self.assertEqual(clean_transcribed_name("John Doe."), "John Doe")
        self.assertEqual(clean_transcribed_name("Alice!"), "Alice")
        self.assertEqual(clean_transcribed_name("  \"Bob Smith\"  "), "Bob Smith")
        self.assertEqual(clean_transcribed_name("Charlie?"), "Charlie")
        self.assertEqual(clean_transcribed_name("David, "), "David")

    def test_pcm16_to_wav_header(self):
        raw_pcm = np.zeros(1600, dtype=np.int16).tobytes()
        wav_bytes = pcm16_to_wav(raw_pcm, sample_rate=16_000, channels=1)
        
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 16_000)
            self.assertEqual(wf.getnframes(), 1600)


class GroqWhisperAdapterTests(unittest.TestCase):
    def test_model_normalization(self):
        adapter = GroqWhisperAdapter(api_key="test_key", model="whisper-v3-turbo")
        self.assertEqual(adapter.model, "whisper-large-v3-turbo")

        adapter2 = GroqWhisperAdapter(api_key="test_key", model="whisper-large-v3-turbo")
        self.assertEqual(adapter2.model, "whisper-large-v3-turbo")

    def test_availability(self):
        adapter_no_key = GroqWhisperAdapter(api_key=None)
        self.assertFalse(adapter_no_key.is_available)

        adapter_with_key = GroqWhisperAdapter(api_key="gsk_valid_key_format")
        self.assertTrue(adapter_with_key.is_available)

    def test_transcribe_mock(self):
        adapter = GroqWhisperAdapter(api_key="gsk_test")
        mock_response = MagicMock()
        mock_response.text = "My name is Alexander Hamilton."

        with patch.object(adapter._client.audio.transcriptions, "create", return_value=mock_response) as mock_create:
            pcm = np.zeros(1600, dtype=np.int16).tobytes()
            result = adapter.transcribe(pcm)

            self.assertEqual(result.provider, "groq")
            self.assertEqual(result.cleaned_name, "Alexander Hamilton")
            self.assertFalse(result.is_fallback)
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            self.assertEqual(kwargs["model"], "whisper-large-v3-turbo")


class FasterWhisperAdapterTests(unittest.TestCase):
    def test_availability(self):
        adapter = FasterWhisperAdapter(model_name_or_path="small")
        self.assertTrue(adapter.is_available)

    def test_transcribe_mock(self):
        adapter = FasterWhisperAdapter(model_name_or_path="small")
        mock_segment = MagicMock()
        mock_segment.text = "Sarah Connor."
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], None)

        with patch.object(adapter, "_load_model", return_value=mock_model):
            pcm = np.zeros(1600, dtype=np.int16).tobytes()
            result = adapter.transcribe(pcm)

            self.assertEqual(result.provider, "faster-whisper")
            self.assertEqual(result.cleaned_name, "Sarah Connor")
            self.assertFalse(result.is_fallback)

    def test_transcribe_real_small_model(self):
        adapter = FasterWhisperAdapter(model_name_or_path="small")
        pcm = np.zeros(16000, dtype=np.int16).tobytes()
        result = adapter.transcribe(pcm)
        self.assertEqual(result.provider, "faster-whisper")
        self.assertEqual(result.model, "small")
        self.assertIsInstance(result.latency_sec, float)


class HybridSpeechTranscriberTests(unittest.TestCase):
    def setUp(self):
        self.mock_groq = MagicMock(spec=GroqWhisperAdapter)
        self.mock_faster = MagicMock(spec=FasterWhisperAdapter)
        self.pcm = np.zeros(1600, dtype=np.int16).tobytes()

    def test_primary_groq_succeeds(self):
        self.mock_groq.is_available = True
        self.mock_groq.transcribe.return_value = TranscriptionResult(
            text="Alice Walker",
            cleaned_name="Alice Walker",
            provider="groq",
            model="whisper-large-v3-turbo",
            latency_sec=0.25,
            is_fallback=False,
        )

        orchestrator = HybridSpeechTranscriber(
            groq_adapter=self.mock_groq,
            faster_whisper_adapter=self.mock_faster,
        )

        result = orchestrator.transcribe(self.pcm)
        self.assertEqual(result.provider, "groq")
        self.assertEqual(result.cleaned_name, "Alice Walker")
        self.assertFalse(result.is_fallback)
        self.assertFalse(orchestrator.fallback_occurred)
        self.mock_faster.transcribe.assert_not_called()

    def test_groq_network_error_falls_back_to_faster_whisper(self):
        self.mock_groq.is_available = True
        self.mock_groq.transcribe.side_effect = ConnectionError("Network unreachable / connection timeout")

        self.mock_faster.is_available = True
        self.mock_faster.transcribe.return_value = TranscriptionResult(
            text="Bob Dylan",
            cleaned_name="Bob Dylan",
            provider="faster-whisper",
            model="small",
            latency_sec=1.1,
            is_fallback=False,
        )

        orchestrator = HybridSpeechTranscriber(
            groq_adapter=self.mock_groq,
            faster_whisper_adapter=self.mock_faster,
        )

        result = orchestrator.transcribe(self.pcm)
        self.assertEqual(result.provider, "faster-whisper")
        self.assertEqual(result.cleaned_name, "Bob Dylan")
        self.assertTrue(result.is_fallback)
        self.assertTrue(orchestrator.fallback_occurred)
        self.assertIn("Groq primary failed", orchestrator.last_error)

    def test_groq_not_configured_uses_faster_whisper(self):
        self.mock_groq.is_available = False
        self.mock_faster.is_available = True
        self.mock_faster.transcribe.return_value = TranscriptionResult(
            text="Charles Babbage",
            cleaned_name="Charles Babbage",
            provider="faster-whisper",
            model="small",
            latency_sec=0.8,
            is_fallback=False,
        )

        orchestrator = HybridSpeechTranscriber(
            groq_adapter=self.mock_groq,
            faster_whisper_adapter=self.mock_faster,
        )

        result = orchestrator.transcribe(self.pcm)
        self.assertEqual(result.provider, "faster-whisper")
        self.assertEqual(result.cleaned_name, "Charles Babbage")
        self.assertTrue(result.is_fallback)  # fallback since Groq was not primary

    def test_both_fail_returns_manual_fallback(self):
        self.mock_groq.is_available = True
        self.mock_groq.transcribe.side_effect = TimeoutError("Groq timeout")
        self.mock_faster.is_available = True
        self.mock_faster.transcribe.side_effect = RuntimeError("faster-whisper out of memory")

        orchestrator = HybridSpeechTranscriber(
            groq_adapter=self.mock_groq,
            faster_whisper_adapter=self.mock_faster,
        )

        result = orchestrator.transcribe(self.pcm)
        self.assertEqual(result.provider, "manual")
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.cleaned_name, "")
        self.assertIsNotNone(result.error)


class OptionalAudioInputTests(unittest.TestCase):
    def test_optional_audio_input_contract(self):
        audio = OptionalAudioInput()
        level, texts, status = audio.poll()
        self.assertIsInstance(level, float)
        self.assertIsInstance(texts, list)
        self.assertTrue(hasattr(audio, "available"))
        self.assertTrue(hasattr(audio, "stt_available"))
        self.assertTrue(audio.stt_available)

    def test_process_utterance_emits_transcript(self):
        mock_transcriber = MagicMock(spec=HybridSpeechTranscriber)
        mock_transcriber.has_any_stt = True
        mock_transcriber.get_initial_status.return_value = "Mock STT active."
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            text="Ada Lovelace",
            cleaned_name="Ada Lovelace",
            provider="groq",
            model="whisper-large-v3-turbo",
            latency_sec=0.2,
            is_fallback=False,
        )

        audio = OptionalAudioInput(transcriber=mock_transcriber)
        # Push 5 chunks of 1600 samples (3200 bytes)
        blocks = [b"\x00\x01" * 1600 for _ in range(5)]
        audio._process_utterance(blocks, min_blocks=3)

        level, texts, status = audio.poll()
        self.assertEqual(texts, ["Ada Lovelace"])
        self.assertIn("Ada Lovelace", status)
        self.assertIn("Groq", status)


class VoiceRecognitionScreenParsingTests(unittest.TestCase):
    def test_parsed_name_cleans_conversational_and_punctuation(self):
        screen = VoiceRecognitionScreen()
        screen.transcript = "My name is Grace Hopper."
        self.assertEqual(screen._parsed_name(), "Grace Hopper")

        screen.transcript = "I am Alan Turing!"
        self.assertEqual(screen._parsed_name(), "Alan Turing")

        screen.transcript = "  Katherine Johnson  "
        self.assertEqual(screen._parsed_name(), "Katherine Johnson")

    def test_manual_typing_prevents_voice_overwrite(self):
        screen = VoiceRecognitionScreen()
        self.assertFalse(screen._user_edited_transcript)
        
        # User types manually
        screen.set_transcript_from_input("Manual Name")
        self.assertTrue(screen._user_edited_transcript)
        self.assertEqual(screen.transcript, "Manual Name")


if __name__ == "__main__":
    unittest.main()
