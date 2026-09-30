#!/usr/bin/env python3
"""PC microphone companion for the existing LS-Face enrollment screen.

Open the enrollment name screen in app/main.py on the Pi, then run:
    python scripts/laptop_voice_companion.py --host <pi-ip> --sttloc pi
The Pi transcribes audio; the PC previews/edits/confirms the proposed name.
No camera or enrollment database is opened by this companion.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import wave

import numpy as np

# Add project root to sys.path so engine modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))

from app.src.engine.speech_transcriber import (
    FasterWhisperAdapter,
    GroqWhisperAdapter,
    HybridSpeechTranscriber,
    TranscriptionResult,
    clean_transcribed_name,
    pcm16_to_wav,
)

class LaptopVoiceCompanion:
    """Manages audio recording on laptop, speech-to-text, and transmission to Raspberry Pi."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5055,
        groq_api_key: str | None = None,
        timeout: float = 4.0,
        stt_loc: str = 'pc'
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.stt_loc = stt_loc
        self.sample_rate = 16_000
        self.transcriber = None
        # Initialize speech transcriber
        if self.stt_loc == 'pc':
            groq_adapter = GroqWhisperAdapter(api_key=groq_api_key)
            faster_adapter = FasterWhisperAdapter(model_name_or_path="small")
            self.transcriber = HybridSpeechTranscriber(
                groq_adapter=groq_adapter,
                faster_whisper_adapter=faster_adapter,
            )

    def check_connection(self) -> tuple[bool, str]:
        """Verify reachability of the LS-Face receiver on Raspberry Pi."""
        url = f"{self.base_url}/health"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LSFaceLaptopCompanion/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return True, f"Connected to {data.get('device', 'Raspberry Pi')} ({data.get('service', 'LS-Face')})"
                return False, f"Unexpected response status: {resp.status}"
        except urllib.error.URLError as exc:
            return False, f"Could not reach {url}: {exc.reason}"
        except Exception as exc:
            return False, f"Connection error: {exc}"

    def transfer_name(
        self,
        name: str,
        provider: str = "groq",
        model: str = "whisper-large-v3-turbo",
    ) -> tuple[bool, str]:
        """Send the transcribed or typed name to the Raspberry Pi."""
        cleaned = clean_transcribed_name(name)
        if not cleaned:
            return False, "Name cannot be empty."

        url = f"{self.base_url}/api/name"
        payload = json.dumps(
            {
                "name": cleaned,
                "provider": provider,
                "model": model,
                "source": "laptop",
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "LSFaceLaptopCompanion/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return True, f"Name '{cleaned}' successfully sent to Raspberry Pi!"
                return False, f"Server responded with status {resp.status}"
        except urllib.error.URLError as exc:
            return False, f"Failed to transfer name to {url}: {exc.reason}"
        except Exception as exc:
            return False, f"Transfer error: {exc}"

    def record_audio(self, duration_sec: float = 3.5) -> bytes | None:
        """Record audio from the laptop's microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[WARN] sounddevice package not installed. Cannot access laptop microphone.")
            return None
        except Exception as exc:
            print(f"[WARN] Audio recording unavailable on laptop: {exc}")
            return None

        total_frames = int(duration_sec * self.sample_rate)
        print(f"\n[RECORDING] Listening on laptop microphone for {duration_sec:.1f}s... (Speak your name now)")

        try:
            recording = sd.rec(
                total_frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            pcm_bytes = recording.tobytes()
            print("[RECORDING] Capture complete.")
            return pcm_bytes
        except Exception as exc:
            print(f"[ERROR] Failed recording from laptop microphone: {exc}")
            return None

    def upload_audio(self, pcm_bytes: bytes) -> dict:
        """ Send recorded audio to the Pi and return its pending result. """
        wav_bytes = pcm16_to_wav(pcm_bytes, sample_rate=self.sample_rate)
        request = urllib.request.Request(
            f"{self.base_url}/api/transcribe",
            data=wav_bytes,
            headers={'Content-Type': 'audio/wav'},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))

        return result

    def propose_name(self, name):
        with urllib.request.urlopen(
            f"{self.base_url}/api/status", timeout=self.timeout
        ) as response:
            status = json.loads(response.read().decode("utf-8"))
        payload = json.dumps({
            "name": name,
            "enrollment_session_id": status.get("enrollment_session_id"),
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/name", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def wait_for_enrollment(self, enrollment_id: str) -> None:
        url = f"{self.base_url}/api/enrollment/{enrollment_id}"
        deadline = time.monotonic() + 120

        while time.monotonic() < deadline:
            with urllib.request.urlopen(
                url, timeout=self.timeout
            ) as response:
                result = json.loads(response.read().decode("utf-8"))

            state = result.get("state")

            if state == "complete":
                print("[ENROLLED] Face enrollment saved successfully.")
                return

            if state == "awaiting_name":
                print("[NOT ACCEPTED] The app rejected the name; retry or use its name field.")
                return

            if state in ("failed", "cancelled"):
                print(f"[ENROLLMENT {state.upper()}] Start a new enrollment.")
                return

            time.sleep(0.5)

        print(
            "[STATUS UNKNOWN] Enrollment may still be running. "
            "Check the receiver before starting another enrollment."
        )

    def cancel_enrollment(self, enrollment_id: str) -> None:
        request = urllib.request.Request(
            f"{self.base_url}/api/enrollment/{enrollment_id}/cancel",
            data=b"",
            method="POST",
        )

        with urllib.request.urlopen(
            request, timeout=self.timeout
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

        print(f"[ENROLLMENT] {result.get('state')}")

    def confirm_name(self, req_id: str, name: str) -> dict:
        """ Send an explicitly accepted name to the Pi. """
        payload = json.dumps({
            'request_id': req_id,
            'name': name
        }).encode('utf-8')

        request = urllib.request.Request(
            f"{self.base_url}/api/confirm",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode('utf-8'))

    def transcribe_audio(self, pcm_bytes: bytes) -> TranscriptionResult:
        """Transcribe PCM bytes using Groq -> faster-whisper -> manual."""
        return self.transcriber.transcribe(pcm_bytes, sample_rate=self.sample_rate)

    def interactive_session(self, default_duration: float = 3.5) -> None:
        print(f"Target receiver: {self.base_url}")

        ok, message = self.check_connection()
        print(f"[CONNECTION] {message}")
        if not ok:
            return

        if self.stt_loc == "pi":
            print("Press Enter to record, type a name, or type 'check' / 'q'.")
            print("You can edit the proposed name before accepting.")
        else:
            print(self.transcriber.get_initial_status())
            print("Press Enter to record, type a name, or type 'q'.")

        while True:
            try:
                command = input("Voice input: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                return

            if command.lower() in ("q", "quit", "exit"):
                return

            if command.lower() == "check":
                ok, message = self.check_connection()
                print(f"[CONNECTION] {message}")
                continue

            if command and self.stt_loc == "pc":
                ok, message = self.transfer_name(command, provider="laptop-manual")
                print(message)
                continue

            pcm = None if command else self.record_audio(duration_sec=default_duration)
            if not pcm and not command:
                command = input("Recording unavailable. Type a name (blank to retry): ").strip()
                if not command:
                    continue
                if self.stt_loc == "pc":
                    ok, message = self.transfer_name(command, provider="laptop-manual")
                    print(message)
                    continue

            if self.stt_loc == "pc":
                result = self.transcribe_audio(pcm)
                if result.cleaned_name:
                    ok, message = self.transfer_name(
                        result.cleaned_name,
                        provider=result.provider,
                        model=result.model,
                    )
                    print(message)
                else:
                    print(f"[ERROR] {result.error or 'No name detected.'}")
                continue

            enrollment_id = None

            try:
                print("[TRANSCRIBING] Sending audio to receiver...")
                pending = self.propose_name(command) if command else self.upload_audio(pcm)

                if not isinstance(pending, dict):
                    raise ValueError("Receiver returned an invalid response.")

                name = pending.get("proposed_name")
                request_id = pending.get("request_id")
                enrollment_id = pending.get("enrollment_session_id")

                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (name, request_id, enrollment_id)
                ):
                    raise ValueError(
                        "Receiver response is missing the name, request ID, "
                        "or enrollment session ID."
                    )

                print(f"[RESULT] Proposed name: {name}")
                print(
                    f"[PROVIDER] {pending.get('provider')} "
                    f"({pending.get('model')})"
                )

                while True:
                    choice = input(
                        "[A] Accept / [E] Edit / [R] Re-record / [C] Cancel: "
                    ).strip().lower()

                    if choice == "a":
                        confirmed = self.confirm_name(request_id, name)

                        if confirmed.get("status") != "queued":
                            print("[ERROR] Receiver did not confirm the name.")
                            continue

                        print(f"[CONFIRMATION SENT] {name}")
                        self.wait_for_enrollment(enrollment_id)
                        break

                    elif choice == "e":
                        edited = input("Correct name: ").strip()
                        if edited:
                            name = edited
                            print(f"[RESULT] Proposed name: {name}")
                        else:
                            print("[ERROR] Name cannot be empty.")

                    elif choice == "r":
                        print("Press Enter at the next prompt to record again.")
                        break

                    elif choice == "c":
                        self.cancel_enrollment(enrollment_id)
                        print('Start a new enrollment')
                        break

                    else:
                        print("Choose A, E, R, or C.")

            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                print(f"[HTTP {exc.code}] {details}")
                if enrollment_id:
                    print(
                        "Check enrollment status on the receiver "
                        "before retrying confirmation."
                    )
                    return

            except (urllib.error.URLError, ValueError, OSError) as exc:
                print(f"[ERROR] {exc}")
                if enrollment_id:
                    print(
                        "The enrollment outcome may be unknown. "
                        "Check the receiver before restarting."
                    )
                    return

            except (KeyboardInterrupt, EOFError):
                if enrollment_id:
                    try:
                        self.cancel_enrollment(enrollment_id)
                    except Exception as exc:
                        print(
                            f"[CANCEL NOT CONFIRMED] {exc}. "
                            "Check the receiver."
                        )
                return


def parse_args():
    parser = argparse.ArgumentParser(description="Laptop Voice Companion for LS-Face Raspberry Pi")
    parser.add_argument(
        "--host",
        "-H",
        default=os.environ.get("LSFACE_HOST", "localhost"),
        help="Raspberry Pi IP address or hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.environ.get("LSFACE_REMOTE_VOICE_PORT", "5055")),
        help="Raspberry Pi remote voice receiver port (default: 5055)",
    )
    parser.add_argument(
        "--groq-key",
        default=os.environ.get("GROQ_API_KEY"),
        help="Groq API Key (reads GROQ_API_KEY env by default)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=3.5,
        help="Recording duration in seconds (default: 3.5)",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Directly send a typed identity name without recording",
    )
    parser.add_argument(
        "--audio-file",
        "-f",
        help="Path to WAV audio file to transcribe and send",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Record once, transcribe, transfer, and exit",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connection to Raspberry Pi and exit",
    )
    parser.add_argument(
        '--sttloc',
        choices=('pi', 'pc'),
        default='pi',
        help="Device where transcription runs: pc or pi (default: pi)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.sttloc == "pi" and (args.audio_file or args.once or args.text is not None):
        raise SystemExit("Pi mode uses interactive recording or typed input; omit --once, --audio-file and --text.")
    companion = LaptopVoiceCompanion(
        host=args.host,
        port=args.port,
        groq_api_key=args.groq_key,
        stt_loc=args.sttloc
    )

    if args.check:
        ok, msg = companion.check_connection()
        print(f"[CHECK] {'OK' if ok else 'FAIL'} - {msg}")
        sys.exit(0 if ok else 1)

    if args.text:
        ok, msg = companion.transfer_name(args.text, provider="laptop-cli")
        print(f"[{'SUCCESS' if ok else 'FAILED'}] {msg}")
        sys.exit(0 if ok else 1)

    if args.audio_file:
        audio_path = Path(args.audio_file)
        if not audio_path.is_file():
            print(f"[ERROR] Audio file not found: {audio_path}")
            sys.exit(1)
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        result = companion.transcribe_audio(frames)
        print(f"[RESULT] Transcribed '{result.cleaned_name}' via {result.provider}")
        ok, msg = companion.transfer_name(result.cleaned_name, provider=result.provider, model=result.model)
        print(f"[{'SUCCESS' if ok else 'FAILED'}] {msg}")
        sys.exit(0 if ok else 1)

    if args.once:
        pcm = companion.record_audio(duration_sec=args.duration)
        if not pcm:
            print("[ERROR] Recording failed.")
            sys.exit(1)
        result = companion.transcribe_audio(pcm)
        if not result.cleaned_name:
            print(f"[ERROR] Transcription failed: {result.error}")
            sys.exit(1)
        ok, msg = companion.transfer_name(result.cleaned_name, provider=result.provider, model=result.model)
        print(f"[{'SUCCESS' if ok else 'FAILED'}] {msg}")
        sys.exit(0 if ok else 1)

    # Default: interactive loop
    companion.interactive_session(default_duration=args.duration)


if __name__ == "__main__":
    main()
