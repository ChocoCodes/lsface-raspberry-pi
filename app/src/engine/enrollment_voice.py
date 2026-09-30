"""In-memory PC speech input for the existing enrollment screen.

No camera, feature extraction, database writes, or Kivy imports belong here.
The caller supplies a UI-thread dispatcher and enrollment callbacks.
"""
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4


class VoiceError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class EnrollmentVoiceService:
    def __init__(self, dispatch, accept, cancel, transcriber=None):
        self.dispatch, self.accept, self.cancel = dispatch, accept, cancel
        self.transcriber = transcriber
        self.lock = threading.RLock()
        self.decode_lock = threading.Lock()
        self.session_id = None
        self.state = "idle"
        self.pending = None
        self.server = None

    def begin(self, session_id):
        with self.lock:
            self.session_id, self.state, self.pending = session_id, "awaiting_name", None

    def update(self, session_id, state):
        with self.lock:
            if session_id == self.session_id:
                self.state = state
                if state in ("complete", "failed", "cancelled"):
                    self.pending = None

    def status(self, session_id=None):
        with self.lock:
            if session_id is not None and session_id != self.session_id:
                raise VoiceError(404, "Enrollment no longer available")
            return {"enrollment_session_id": self.session_id, "state": self.state}

    def _active(self, session_id):
        if session_id != self.session_id or self.state != "awaiting_name":
            raise VoiceError(409, "Open an active enrollment name screen first")

    def propose(self, name, provider="manual", model="", session_id=None):
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise VoiceError(400, "Name must contain 1 to 200 characters")
        with self.lock:
            self._active(session_id)
            self.pending = {
                "request_id": uuid4().hex, "enrollment_session_id": session_id,
                "status": "pending", "proposed_name": name.strip(),
                "provider": provider, "model": model,
            }
            return dict(self.pending)

    def transcribe(self, body):
        with self.lock:
            session_id = self.session_id
            self._active(session_id)
        try:
            with wave.open(io.BytesIO(body), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
                    raise ValueError("Expected mono 16-bit 16 kHz PCM WAV")
                count = wav.getnframes()
                if not 0 < count <= 240000:
                    raise ValueError("Recording must be at most 15 seconds")
                pcm = wav.readframes(count)
                if len(pcm) != count * 2:
                    raise ValueError("Truncated WAV")
        except (ValueError, wave.Error, EOFError) as exc:
            raise VoiceError(400, str(exc)) from exc
        if not self.decode_lock.acquire(blocking=False):
            raise VoiceError(503, "Transcriber busy; retry shortly")
        try:
            if self.transcriber is None:
                from .speech_transcriber import HybridSpeechTranscriber
                self.transcriber = HybridSpeechTranscriber()
            result = self.transcriber.transcribe(pcm, sample_rate=16000)
        except Exception as exc:
            raise VoiceError(503, "Transcription failed; retry or type a name") from exc
        finally:
            self.decode_lock.release()
        if not result.cleaned_name:
            raise VoiceError(422, "No name detected; retry or type a name")
        return self.propose(result.cleaned_name, result.provider, result.model, session_id)

    def confirm(self, request_id, name):
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise VoiceError(400, "Name must contain 1 to 200 characters")
        name = name.strip()
        with self.lock:
            pending = self.pending
            if pending is None or pending["request_id"] != request_id:
                raise VoiceError(409, "Recording expired; request a new name")
            if pending["status"] == "queued":
                if pending["confirmed_name"] != name:
                    raise VoiceError(409, "A different name was already confirmed")
                return {"status": "queued"}
            self._active(pending["enrollment_session_id"])
            session_id = self.session_id
            pending.update(status="queued", confirmed_name=name)
            self.state = "confirming"

        def apply():
            with self.lock:
                if self.session_id != session_id or self.state != "confirming":
                    return
            try:
                accepted = self.accept(session_id, name)
            except Exception:
                accepted = False
            with self.lock:
                if self.session_id == session_id and self.state == "confirming":
                    self.state = "confirmed" if accepted else "awaiting_name"
                    if not accepted:
                        self.pending = None
        self.dispatch(apply)
        return {"status": "queued"}

    def request_cancel(self, session_id):
        with self.lock:
            if session_id != self.session_id or self.state not in ("awaiting_name", "confirming", "confirmed"):
                raise VoiceError(409, "Enrollment is saving or no longer active")
            self.state, self.pending = "cancelling", None

        def apply():
            with self.lock:
                if self.session_id != session_id or self.state != "cancelling":
                    return
            if self.cancel(session_id):
                self.update(session_id, "cancelled")
        self.dispatch(apply)
        return {"state": "cancelling"}

    def start(self, host="0.0.0.0", port=5055):
        if self.server is not None:
            return
        self.server = ThreadingHTTPServer((host, port), EnrollmentVoiceHandler)
        self.server.voice = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        with self.lock:
            self.pending = None
            self.state = "cancelled"
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None


class EnrollmentVoiceHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            if self.path in ("/health", "/api/status"):
                data = self.server.voice.status()
                data.update(status="online", device="raspberry-pi", service="lsface-voice-receiver")
            elif self.path.startswith("/api/enrollment/"):
                data = self.server.voice.status(self.path.removeprefix("/api/enrollment/"))
            else:
                raise VoiceError(404, "Not found")
            self.respond(200, data)
        except VoiceError as exc:
            self.respond(exc.status, {"error": str(exc)})

    def do_POST(self):
        voice = self.server.voice
        try:
            if self.path.startswith("/api/enrollment/") and self.path.endswith("/cancel"):
                token = self.path[len("/api/enrollment/"):-len("/cancel")]
                self.respond(202, voice.request_cancel(token))
                return
            if self.path not in ("/api/transcribe", "/api/name", "/api/confirm"):
                raise VoiceError(404, "Not found")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1048576:
                raise VoiceError(413, "Request must contain 1 to 1048576 bytes")
            self.connection.settimeout(10)
            body = self.rfile.read(length)
            if len(body) != length:
                raise VoiceError(400, "Incomplete request")
            if self.path == "/api/transcribe":
                if self.headers.get_content_type() != "audio/wav":
                    raise VoiceError(415, "Expected audio/wav")
                result = voice.transcribe(body)
            else:
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise VoiceError(400, "Expected JSON object")
                if self.path == "/api/confirm":
                    result = voice.confirm(payload.get("request_id"), payload.get("name"))
                else:
                    result = voice.propose(payload.get("name"), session_id=payload.get("enrollment_session_id"))
            self.respond(200, result)
        except VoiceError as exc:
            self.respond(exc.status, {"error": str(exc)})
        except (ValueError, UnicodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except TimeoutError:
            self.respond(408, {"error": "Upload timed out"})
