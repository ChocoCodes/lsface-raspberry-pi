"""Lightweight HTTP server for receiving remote audio or transcribed names.

Allows a companion device (e.g. laptop with microphone) to capture audio,
run Groq whisper-v3-turbo transcription (with local fallback), and transmit
the resulting identity name directly to the Raspberry Pi running LS-Face.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5055


class RemoteAudioHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for remote audio / name reception."""

    server: "RemoteAudioServer"

    def do_GET(self) -> None:
        """Handle health and status check requests."""
        if self.path in ("/health", "/status", "/api/status", "/"):
            self._send_json(
                200,
                {
                    "status": "online",
                    "device": "raspberry-pi",
                    "service": "lsface-voice-receiver",
                    "listening": True,
                },
            )
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        """Handle incoming name payload or raw audio from laptop."""
        if self.path in ("/api/name", "/name", "/api/transcribe"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {"error": "Empty request body"})
                return

            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")

            # Case A: JSON body containing pre-transcribed name
            if "application/json" in content_type or body.startswith(b"{"):
                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception as exc:
                    self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
                    return

                raw_name = payload.get("name") or payload.get("transcript") or ""
                provider = payload.get("provider", "laptop")
                model = payload.get("model", "")

                if not str(raw_name).strip():
                    self._send_json(400, {"error": "No name provided in payload"})
                    return

                # Forward name to the audio engine callback
                self.server.notify_name(str(raw_name).strip(), provider=provider, model=model)

                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "name": str(raw_name).strip(),
                        "provider": provider,
                        "message": "Name received and forwarded to enrollment screen",
                    },
                )
                return

            # Case B: Raw audio/wav payload (if laptop streams raw PCM/WAV to Pi for STT)
            elif "audio" in content_type or "octet-stream" in content_type:
                transcriber = getattr(self.server, "transcriber", None)
                if transcriber is None:
                    self._send_json(503, {"error": "Pi transcriber not available"})
                    return

                try:
                    # Strip WAV header if needed or let transcriber handle
                    result = transcriber.transcribe(body)
                    if result.cleaned_name:
                        self.server.notify_name(
                            result.cleaned_name,
                            provider=result.provider,
                            model=result.model,
                        )
                        self._send_json(
                            200,
                            {
                                "status": "ok",
                                "name": result.cleaned_name,
                                "provider": result.provider,
                                "model": result.model,
                            },
                        )
                    else:
                        self._send_json(
                            200,
                            {
                                "status": "empty",
                                "name": "",
                                "error": result.error or "No speech detected in audio.",
                            },
                        )
                except Exception as exc:
                    self._send_json(500, {"error": f"Transcription error: {exc}"})
                return

            else:
                self._send_json(415, {"error": f"Unsupported Media Type: {content_type}"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        """Send formatted JSON HTTP response with CORS headers."""
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            logger.debug("Failed sending JSON response: %s", exc)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default stderr log spam from BaseHTTPRequestHandler."""
        logger.debug("%s - " + format, self.address_string(), *args)


class RemoteAudioServer(HTTPServer):
    """HTTP server with callback for remote name notifications."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        on_name_received: Optional[Callable[[str, str, str], None]] = None,
        transcriber: Optional[Any] = None,
    ) -> None:
        self.allow_reuse_address = True
        self.on_name_received = on_name_received
        self.transcriber = transcriber
        super().__init__((host, port), RemoteAudioHandler)

    def notify_name(self, name: str, provider: str = "laptop", model: str = "") -> None:
        if self.on_name_received is not None:
            try:
                self.on_name_received(name, provider, model)
            except Exception as exc:
                logger.error("Error invoking on_name_received callback: %s", exc)


class RemoteAudioReceiver:
    """Threaded manager for starting and stopping the RemoteAudioServer."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        on_name_received: Optional[Callable[[str, str, str], None]] = None,
        transcriber: Optional[Any] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_name_received = on_name_received
        self.transcriber = transcriber
        self._server: Optional[RemoteAudioServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running and self._server is not None

    def start(self) -> bool:
        """Start the remote receiver daemon thread."""
        with self._lock:
            if self._running:
                return True

            try:
                self._server = RemoteAudioServer(
                    host=self.host,
                    port=self.port,
                    on_name_received=self.on_name_received,
                    transcriber=self.transcriber,
                )
            except OSError as exc:
                logger.warning(
                    "Could not bind RemoteAudioServer on %s:%d (%s). Trying alternate port.",
                    self.host,
                    self.port,
                    exc,
                )
                try:
                    self.port = self.port + 1
                    self._server = RemoteAudioServer(
                        host=self.host,
                        port=self.port,
                        on_name_received=self.on_name_received,
                        transcriber=self.transcriber,
                    )
                except Exception as err:
                    logger.error("Failed to start RemoteAudioServer: %s", err)
                    self._server = None
                    return False

            self._running = True
            self._thread = threading.Thread(
                target=self._run_server,
                name=f"remote-audio-receiver-{self.port}",
                daemon=True,
            )
            self._thread.start()
            logger.info("RemoteAudioReceiver listening on %s:%d", self.host, self.port)
            return True

    def _run_server(self) -> None:
        try:
            if self._server:
                self._server.serve_forever()
        except Exception as exc:
            logger.debug("RemoteAudioServer stopped: %s", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        """Shutdown the remote receiver server."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            server, self._server = self._server, None
            if server:
                try:
                    server.shutdown()
                    server.server_close()
                except Exception:
                    pass
            thread, self._thread = self._thread, None
            if thread and thread is not threading.current_thread():
                thread.join(timeout=0.5)
