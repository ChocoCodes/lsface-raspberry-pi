from __future__ import annotations

import re
import threading
from uuid import uuid4

import numpy as np

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.screenmanager import Screen

from src.config.config import KV_PATH
from src.engine.audio_input import OptionalAudioInput
from src.engine.database.database_manager import DatabaseManager
from src.engine.database.feature_db import FeatureDB


Builder.load_file(str(KV_PATH / "voice_recognition.kv"))


class VoiceRecognitionScreen(Screen):
    """Name confirmation screen between pose capture and live recognition."""

    prompt_text = StringProperty("Good day! What is your name?")
    transcript = StringProperty("")
    status_text = StringProperty("Waiting for the pose scan…")
    progress_text = StringProperty("Facial features will be prepared here.")
    audio_status = StringProperty("Microphone is optional; type your name below.")
    action_text = StringProperty("Continue")
    state = StringProperty("idle")
    session_id = StringProperty("")
    database_id = StringProperty("")
    enrollment_progress = NumericProperty(0.0)
    audio_level = NumericProperty(0.0)
    continue_enabled = BooleanProperty(False)
    name_locked = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_session_id = None
        self._frames = None
        self._prepared_features = None
        self._prepare_thread = None
        self._commit_thread = None
        self._continue_requested = False
        self._setting_transcript = False
        self._user_edited_transcript = False
        self._audio = None
        self._audio_poll = None

    def on_enter(self, *_args):
        if self._active_session_id is not None:
            self._start_audio()
            self._refresh_gate()
            Clock.schedule_once(self._focus_name_input, 0)

    def on_leave(self, *_args):
        self._invalidate_session()
        self._stop_audio()

    def start_enrollment(
        self,
        frames: dict[str, np.ndarray],
        *,
        database_id: str,
        session_id: str | None = None,
        initial_name: str = "",
    ) -> None:
        """Start non-mutating preparation before showing the name gate."""

        self._invalidate_session()
        self._stop_audio()
        token = session_id or uuid4().hex
        self._active_session_id = token
        self.session_id = token
        self.database_id = str(database_id or "").strip()
        self._frames = {
            str(label): np.array(frame, copy=True) for label, frame in frames.items()
        }
        self._prepared_features = None
        self._continue_requested = False
        self.name_locked = False
        self._set_transcript(" ".join(str(initial_name).split()))
        self._user_edited_transcript = bool(initial_name.strip())
        self.state = "preparing"
        self.enrollment_progress = 0.0
        self.audio_level = 0.0
        self.status_text = "Preparing your facial enrollment…"
        self.progress_text = "Extracting features in the background."
        self.audio_status = "Microphone is optional; type your name below."
        self.action_text = "Continue"
        self._refresh_gate()

        worker = threading.Thread(
            target=self._prepare_worker,
            args=(token, self._frames, self.database_id),
            name="enrollment-feature-preparation",
            daemon=True,
        )
        self._prepare_thread = worker
        worker.start()
        manager = self.manager
        if manager is not None and manager.current == self.name:
            self._start_audio()

    def continue_enrollment(self) -> None:
        """Handle both the Enter key and the visible Continue button."""

        if self._active_session_id is None or self.state == "committing":
            return
        if self.state == "error":
            if self.action_text == "Back":
                self.go_back()
                return
            if self._prepared_features is None:
                self._restart_preparation()
            else:
                try:
                    name = self._parsed_name()
                except ValueError as exc:
                    self.status_text = str(exc)
                    self._refresh_gate()
                    return
                self._set_transcript(name)
                self.name_locked = True
                self._continue_requested = True
                self._begin_commit(name)
            return
        if self.state == "complete":
            return

        try:
            name = self._parsed_name()
        except ValueError as exc:
            self.status_text = str(exc)
            self._refresh_gate()
            return

        self._set_transcript(name)
        self.name_locked = True
        self._continue_requested = True
        self.continue_enabled = False
        if self.state == "preparing":
            self.status_text = "Name captured. Finishing enrollment preparation…"
            self.action_text = "Preparing…"
            return
        if self.state == "ready":
            self._begin_commit(name)

    def set_transcript_from_input(self, value: str) -> None:
        if not self._setting_transcript:
            self._user_edited_transcript = True
        self.transcript = value
        self._refresh_gate()

    def _focus_name_input(self, _dt) -> None:
        if self._active_session_id is None or self.state == "complete":
            return
        name_input = self.ids.get("name_input")
        if name_input is not None and not name_input.disabled:
            name_input.focus = True

    def go_back(self):
        if self.state == "committing":
            return
        self._invalidate_session()
        self._stop_audio()
        self.manager.current = "home"

    def _prepare_worker(self, token: str, frames: dict[str, np.ndarray], database_id: str) -> None:
        prepared = None
        error = None
        try:
            prepare = getattr(DatabaseManager, "prepare_enrollment", None)
            if prepare is None:
                raise RuntimeError(
                    "DatabaseManager.prepare_enrollment(frames, database_id=...) "
                    "is not available yet."
                )
            prepared = prepare(frames, database_id=database_id)
        except Exception as exc:
            error = exc
        Clock.schedule_once(
            lambda _dt: self._finish_preparation(token, prepared, error),
            0,
        )

    def _finish_preparation(self, token: str, prepared, error) -> None:
        if token != self._active_session_id:
            return
        self._prepare_thread = None
        if error is not None:
            self.state = "error"
            self.enrollment_progress = 0.0
            self.status_text = "Enrollment preparation could not finish."
            self.progress_text = str(error)
            self.action_text = "Retry preparation"
            self.name_locked = False
            self._continue_requested = False
            self._refresh_gate()
            return
        if prepared is None:
            self._finish_preparation(
                token,
                None,
                RuntimeError("Enrollment preparation returned no feature data."),
            )
            return

        self._prepared_features = prepared
        self.enrollment_progress = 1.0
        self.state = "ready"
        self.progress_text = "Facial features ready."
        self.status_text = "Enter a name, then press Enter or Continue."
        self.action_text = "Continue"
        if self._continue_requested and self.name_locked:
            try:
                self._begin_commit(self._parsed_name())
            except ValueError as exc:
                self.name_locked = False
                self._continue_requested = False
                self.status_text = str(exc)
                self._refresh_gate()
        else:
            self._refresh_gate()

    def _begin_commit(self, name: str) -> None:
        if self._prepared_features is None or self._commit_thread is not None:
            return
        token = self._active_session_id
        database_id = self.database_id
        prepared = self._prepared_features
        self.state = "committing"
        self.status_text = "Saving enrollment and publishing live recognition…"
        self.progress_text = "This may take a moment."
        self.action_text = "Saving…"
        self.continue_enabled = False

        def worker():
            published = None
            error = None
            try:
                commit = getattr(DatabaseManager, "commit_enrollment", None)
                if commit is None:
                    raise RuntimeError(
                        "DatabaseManager.commit_enrollment(name, prepared_features, "
                        "database_id=...) is not available yet."
                    )
                published = commit(name, prepared, database_id=database_id)
            except Exception as exc:
                error = exc
            Clock.schedule_once(
                lambda _dt: self._finish_commit(token, name, published, error),
                0,
            )

        self._commit_thread = threading.Thread(
            target=worker,
            name="enrollment-release-publication",
            daemon=True,
        )
        self._commit_thread.start()

    def _finish_commit(self, token: str, name: str, published, error) -> None:
        if token != self._active_session_id:
            return
        self._commit_thread = None
        if error is not None:
            self.state = "error"
            self.status_text = "Enrollment could not be saved."
            self.progress_text = str(error)
            self.action_text = "Continue"
            self.name_locked = False
            self._continue_requested = False
            self._refresh_gate()
            return

        self.state = "complete"
        self.enrollment_progress = 1.0
        self.status_text = "Enrollment saved. Opening live recognition…"
        self.progress_text = ""
        self.action_text = ""
        self.continue_enabled = False

        try:
            recognition = self.manager.get_screen("recognition")
            recognition.configure_session(
                database_id=self.database_id,
                expected_identity_name=name,
                session_id=token,
            )
            self.manager.current = "recognition"
        except Exception as exc:
            self.state = "error"
            self.status_text = "Enrollment saved, but recognition could not open."
            self.progress_text = str(exc)
            self.action_text = "Back"

    def _restart_preparation(self) -> None:
        if self._frames is None:
            return
        self.start_enrollment(
            self._frames,
            database_id=self.database_id,
            initial_name=self.transcript,
        )

    def _parsed_name(self) -> str:
        transcript = " ".join(self.transcript.split())
        transcript = re.sub(
            r"^(?:my\s+name\s+is|i\s+am|i['’]m|this\s+is)\s+",
            "",
            transcript,
            flags=re.IGNORECASE,
        )
        return FeatureDB.normalize_name(transcript)

    def _set_transcript(self, value: str) -> None:
        self._setting_transcript = True
        try:
            self.transcript = value
        finally:
            self._setting_transcript = False

    def _start_audio(self) -> None:
        if self._audio is not None:
            return
        self._audio = OptionalAudioInput()
        if not self._audio.available:
            self.audio_status = self._audio.reason
            return
        if not self._audio.start():
            self.audio_status = self._audio.reason
            return
        self.audio_status = self._audio.stt_reason
        self._audio_poll = Clock.schedule_interval(self._poll_audio, 1.0 / 20.0)

    def _stop_audio(self) -> None:
        if self._audio_poll is not None:
            self._audio_poll.cancel()
            self._audio_poll = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        self.audio_level = 0.0

    def _poll_audio(self, _dt) -> None:
        if self._audio is None or self._active_session_id is None:
            return
        level, texts, error = self._audio.poll()
        self.audio_level = level
        if error:
            self.audio_status = error
        if self.name_locked or self._user_edited_transcript:
            return
        for text in texts:
            self._set_transcript(text)
            self._refresh_gate()

    def _refresh_gate(self) -> None:
        if self.state == "error":
            self.continue_enabled = (
                self._active_session_id is not None
                and self.action_text != "Back"
                and (self._prepared_features is None or bool(self.transcript.strip()))
            )
            return
        self.continue_enabled = (
            self._active_session_id is not None
            and not self.name_locked
            and self.state in ("preparing", "ready")
            and bool(self.transcript.strip())
        )

    def _invalidate_session(self) -> None:
        self._active_session_id = None
        self._continue_requested = False
