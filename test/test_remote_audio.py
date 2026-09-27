"""Tests for remote audio server, laptop voice companion, and network transfer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

from app.src.engine.audio_input import OptionalAudioInput
from app.src.engine.remote_audio_server import RemoteAudioReceiver, RemoteAudioServer
from scripts.laptop_voice_companion import LaptopVoiceCompanion


class RemoteAudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.received_names = []

    def _on_name(self, name: str, provider: str, model: str):
        self.received_names.append((name, provider, model))

    def test_remote_receiver_lifecycle_and_endpoints(self):
        # Use an ephemeral test port
        receiver = RemoteAudioReceiver(
            host="127.0.0.1",
            port=5110,
            on_name_received=self._on_name,
        )
        self.assertTrue(receiver.start())
        self.assertTrue(receiver.is_running)

        companion = LaptopVoiceCompanion(host="127.0.0.1", port=receiver.port)

        # 1. Check health
        ok, msg = companion.check_connection()
        self.assertTrue(ok)
        self.assertIn("raspberry-pi", msg.lower())

        # 2. Transfer name via HTTP
        send_ok, send_msg = companion.transfer_name(
            "My name is Katherine Johnson",
            provider="groq",
            model="whisper-large-v3-turbo",
        )
        self.assertTrue(send_ok)
        self.assertIn("Katherine Johnson", send_msg)

        # 3. Verify server received cleaned name
        self.assertEqual(len(self.received_names), 1)
        name, provider, model = self.received_names[0]
        self.assertEqual(name, "Katherine Johnson")
        self.assertEqual(provider, "groq")
        self.assertEqual(model, "whisper-large-v3-turbo")

        receiver.stop()
        self.assertFalse(receiver.is_running)

    def test_optional_audio_input_remote_integration(self):
        # Start OptionalAudioInput which starts its remote receiver
        audio = OptionalAudioInput(remote_port=5112)
        self.assertTrue(audio.start())
        self.assertIn("5112", audio.stt_reason)

        companion = LaptopVoiceCompanion(host="127.0.0.1", port=audio.remote_port)
        ok, msg = companion.check_connection()
        self.assertTrue(ok)

        # Send name from simulated companion
        trans_ok, trans_msg = companion.transfer_name(
            "Alan Turing.",
            provider="groq",
            model="whisper-large-v3-turbo",
        )
        self.assertTrue(trans_ok)

        level, texts, status = audio.poll()
        self.assertIn("Alan Turing", texts)
        self.assertIsNotNone(status)
        self.assertIn("Alan Turing", status)
        self.assertIn("laptop", status.lower())

        audio.stop()

    def test_companion_transcribe_and_transfer_mock(self):
        companion = LaptopVoiceCompanion(host="127.0.0.1", port=5114)
        mock_result = MagicMock()
        mock_result.cleaned_name = "Grace Hopper"
        mock_result.provider = "groq"
        mock_result.model = "whisper-large-v3-turbo"
        mock_result.is_fallback = False
        mock_result.latency_sec = 0.25

        with patch.object(companion.transcriber, "transcribe", return_value=mock_result):
            res = companion.transcribe_audio(b"dummy_pcm")
            self.assertEqual(res.cleaned_name, "Grace Hopper")

            with patch.object(companion, "transfer_name", return_value=(True, "Success")):
                ok, msg = companion.transfer_name(res.cleaned_name, provider=res.provider, model=res.model)
                self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
