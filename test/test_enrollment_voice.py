"""STT integration tests: no camera, microphone, cloud calls, or database writes."""
import ast
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from src.engine.enrollment_voice import EnrollmentVoiceService, VoiceError
from scripts.laptop_voice_companion import LaptopVoiceCompanion


def wav_bytes(rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


class EnrollmentVoiceTests(unittest.TestCase):
    def setUp(self):
        self.callbacks = []
        self.accept = Mock(return_value=True)
        self.cancel = Mock(return_value=True)
        self.transcriber = Mock()
        self.transcriber.transcribe.return_value = SimpleNamespace(
            cleaned_name="John", provider="groq", model="test-model")
        self.voice = EnrollmentVoiceService(
            self.callbacks.append, self.accept, self.cancel, self.transcriber)
        self.voice.begin("face-session")

    def tearDown(self):
        self.voice.close()

    def proposal(self):
        return self.voice.propose("John", session_id="face-session")

    def test_upload_decodes_wav_without_accepting_name(self):
        result = self.voice.transcribe(wav_bytes())
        self.transcriber.transcribe.assert_called_once_with(b"\0\0" * 1600, sample_rate=16000)
        self.assertEqual(result["enrollment_session_id"], "face-session")
        self.accept.assert_not_called()
        self.assertEqual(self.callbacks, [])

    def test_invalid_wav_never_calls_provider(self):
        for body in (b"invalid", wav_bytes(8000)):
            with self.subTest(body_length=len(body)), self.assertRaises(VoiceError):
                self.voice.transcribe(body)
        self.transcriber.transcribe.assert_not_called()

    def test_edited_confirmation_dispatched_once(self):
        request = self.proposal()
        self.voice.confirm(request["request_id"], "Jane")
        self.voice.confirm(request["request_id"], "Jane")
        self.accept.assert_not_called()
        self.assertEqual(len(self.callbacks), 1)
        self.callbacks.pop()()
        self.accept.assert_called_once_with("face-session", "Jane")
        self.assertEqual(self.voice.status()["state"], "confirmed")

    def test_conflicting_duplicate_is_rejected(self):
        request = self.proposal()
        self.voice.confirm(request["request_id"], "Jane")
        with self.assertRaises(VoiceError):
            self.voice.confirm(request["request_id"], "Other")

    def test_new_session_rejects_old_request_and_queued_callback(self):
        request = self.proposal()
        self.voice.confirm(request["request_id"], "Jane")
        self.voice.begin("next-person")
        self.callbacks.pop()()
        self.accept.assert_not_called()
        with self.assertRaises(VoiceError):
            self.voice.confirm(request["request_id"], "Jane")

    def test_rerecord_replaces_previous_proposal(self):
        old = self.proposal()
        self.proposal()
        with self.assertRaises(VoiceError):
            self.voice.confirm(old["request_id"], "John")

    def test_cancel_clears_name_and_dispatches_ui_navigation(self):
        self.proposal()
        self.voice.request_cancel("face-session")
        self.assertIsNone(self.voice.pending)
        self.cancel.assert_not_called()
        self.callbacks.pop()()
        self.cancel.assert_called_once_with("face-session")
        self.assertEqual(self.voice.status()["state"], "cancelled")

    def test_cancel_prevents_queued_acceptance(self):
        pending = self.proposal()
        self.voice.confirm(pending["request_id"], "John")
        self.voice.request_cancel("face-session")
        for callback in self.callbacks:
            callback()
        self.accept.assert_not_called()

    def test_terminal_states_discard_names_but_preserve_status(self):
        for state in ("complete", "failed", "cancelled"):
            self.voice.begin("face-session")
            self.proposal()
            self.voice.update("face-session", state)
            self.assertIsNone(self.voice.pending)
            self.assertEqual(self.voice.status("face-session")["state"], state)

    def test_cannot_cancel_during_save(self):
        self.voice.update("face-session", "committing")
        with self.assertRaises(VoiceError):
            self.voice.request_cancel("face-session")

    def test_late_transcription_cannot_restore_cancelled_name(self):
        def transcribe(*args, **kwargs):
            self.voice.update("face-session", "cancelled")
            return SimpleNamespace(cleaned_name="John", provider="mock", model="mock")
        self.transcriber.transcribe.side_effect = transcribe
        with self.assertRaises(VoiceError):
            self.voice.transcribe(wav_bytes())
        self.assertIsNone(self.voice.pending)

    def test_provider_failure_allows_manual_input(self):
        self.transcriber.transcribe.side_effect = RuntimeError("offline")
        with self.assertRaises(VoiceError):
            self.voice.transcribe(wav_bytes())
        self.assertEqual(self.proposal()["proposed_name"], "John")

    def test_ui_rejection_allows_retry_without_committing(self):
        self.accept.return_value = False
        pending = self.proposal()
        self.voice.confirm(pending["request_id"], "John")
        self.callbacks.pop()()
        self.assertEqual(self.voice.status()["state"], "awaiting_name")
        self.assertIsNone(self.voice.pending)

    def test_http_companion_upload_confirm_and_status(self):
        self.voice.start("127.0.0.1", 0)
        companion = LaptopVoiceCompanion(
            host="127.0.0.1", port=self.voice.server.server_port, stt_loc="pi")
        self.assertTrue(companion.check_connection()[0])
        proposal = companion.upload_audio(b"\0\0" * 1600)
        self.assertEqual(companion.confirm_name(proposal["request_id"], "John")["status"], "queued")
        self.callbacks.pop()()
        self.voice.update("face-session", "complete")
        companion.wait_for_enrollment("face-session")
        self.assertIsNone(self.voice.pending)

    def test_http_manual_name_cancel_and_invalid_payload(self):
        self.voice.start("127.0.0.1", 0)
        companion = LaptopVoiceCompanion(
            host="127.0.0.1", port=self.voice.server.server_port, stt_loc="pi")
        self.assertEqual(companion.propose_name("Typed Name")["proposed_name"], "Typed Name")
        request = urllib.request.Request(
            companion.base_url + "/api/confirm", data=b"[]",
            headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=2)
        self.assertEqual(error.exception.code, 400)
        companion.cancel_enrollment("face-session")
        self.callbacks.pop()()
        self.assertEqual(self.voice.state, "cancelled")

    def test_screen_adapter_uses_existing_continue_on_ui_dispatch(self):
        # Execute the actual method without initializing Kivy/OpenGL hardware.
        tree = ast.parse((ROOT / "app/src/views/voice_recognition.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "accept_confirmed_name")
        namespace = {"FeatureDB": SimpleNamespace(normalize_name=lambda text: text.strip())}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "screen-adapter", "exec"), namespace)
        screen = SimpleNamespace(_active_session_id="face-session", state="preparing", name_locked=False, _set_transcript=Mock())
        def proceed():
            screen.name_locked = True
        screen.continue_enrollment = Mock(side_effect=proceed)
        self.voice.accept = lambda token, name: namespace["accept_confirmed_name"](screen, token, name)
        proposal = self.proposal()
        self.voice.confirm(proposal["request_id"], "Jane")
        screen.continue_enrollment.assert_not_called()
        self.callbacks.pop()()
        screen._set_transcript.assert_called_once_with("Jane")
        screen.continue_enrollment.assert_called_once()
        self.assertEqual(self.voice.state, "confirmed")

    def screen_method(self, name):
        tree = ast.parse((ROOT / "app/src/views/voice_recognition.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), name, "exec"), scope)
        return scope[name]

    def test_screen_save_failure_discards_data_and_reports_failure(self):
        screen = SimpleNamespace(
            _active_session_id="face-session", _voice_service=self.voice,
            _frames={"FRONT": object()}, _prepared_features=object(),
            _set_transcript=Mock())
        self.proposal()
        self.screen_method("_finish_commit")(
            screen, "face-session", "John", None, RuntimeError("save failed"))
        self.assertIsNone(screen._frames)
        self.assertIsNone(screen._prepared_features)
        screen._set_transcript.assert_called_once_with("")
        self.assertEqual(self.voice.state, "failed")
        self.assertIsNone(self.voice.pending)

    def test_screen_preparation_failure_discards_data(self):
        screen = SimpleNamespace(
            _active_session_id="face-session", _voice_service=self.voice,
            _frames={"FRONT": object()}, _prepared_features=object(),
            _set_transcript=Mock())
        self.screen_method("_finish_preparation")(
            screen, "face-session", None, RuntimeError("extraction failed"))
        self.assertEqual(self.voice.state, "failed")
        self.assertIsNone(screen._frames)
        self.assertIsNone(screen._prepared_features)

    def test_success_still_opens_existing_live_recognition(self):
        recognition = SimpleNamespace(configure_session=Mock())
        home = SimpleNamespace()
        screens = {"home": home, "recognition": recognition}
        screen = SimpleNamespace(
            _active_session_id="face-session", _voice_service=self.voice,
            database_id="selected-db", camera_mode="existing-camera",
            manager=SimpleNamespace(get_screen=screens.__getitem__),
        )
        published = object()
        self.screen_method("_finish_commit")(screen, "face-session", "John", published, None)
        self.assertEqual(self.voice.state, "complete")
        self.assertIs(home.feature_db, published)
        self.assertEqual(screen.manager.current, "recognition")
        recognition.configure_session.assert_called_once_with(
            database_id="selected-db", expected_identity_name="John", session_id="face-session")

    def test_manual_companion_input_requires_accept_and_waits_for_app(self):
        companion = LaptopVoiceCompanion(stt_loc="pi")
        proposal = self.proposal()
        with patch.object(companion, "check_connection", return_value=(True, "online")), patch.object(companion, "propose_name", return_value=proposal) as propose, patch.object(companion, "record_audio") as record, patch.object(companion, "confirm_name", return_value={"status": "queued"}) as confirm, patch.object(companion, "wait_for_enrollment") as wait, patch("builtins.input", side_effect=["John", "e", "Jane", "a"]):
            companion.interactive_session()
        record.assert_not_called()
        propose.assert_called_once_with("John")
        confirm.assert_called_once_with(proposal["request_id"], "Jane")
        wait.assert_called_once_with("face-session")


if __name__ == "__main__":
    unittest.main()
