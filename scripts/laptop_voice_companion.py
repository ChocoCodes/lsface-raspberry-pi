#!/usr/bin/env python3
"""Laptop Voice Companion for LS-Face Raspberry Pi.

This companion script runs on a laptop (the input device) to:
1. Record audio from the laptop's microphone.
2. Transcribe the spoken identity name using:
   - Primary: Groq Whisper API (whisper-large-v3-turbo)
   - Fallback: Local faster-whisper (small model) if network is flaky
   - Fallback: Manual typing if both STT options fail
3. Send the transcribed name over the local network to the Raspberry Pi
   running the LS-Face enrollment screen.

Usage Examples:
    # Interactive recording loop (default):
    python scripts/laptop_voice_companion.py --host 192.168.1.50

    # Single-shot recording and transfer:
    python scripts/laptop_voice_companion.py --host 192.168.1.50 --once

    # Send typed name directly to Raspberry Pi:
    python scripts/laptop_voice_companion.py --host 192.168.1.50 --text "Ada Lovelace"

    # Test connection to Raspberry Pi:
    python scripts/laptop_voice_companion.py --host 192.168.1.50 --check
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

try:
    from app.src.engine.speech_transcriber import (
        FasterWhisperAdapter,
        GroqWhisperAdapter,
        HybridSpeechTranscriber,
        TranscriptionResult,
        clean_transcribed_name,
        load_dotenv,
        pcm16_to_wav,
    )
except ImportError:
    from src.engine.speech_transcriber import (
        FasterWhisperAdapter,
        GroqWhisperAdapter,
        HybridSpeechTranscriber,
        TranscriptionResult,
        clean_transcribed_name,
        load_dotenv,
        pcm16_to_wav,
    )

load_dotenv()


class LaptopVoiceCompanion:
    """Manages audio recording on laptop, speech-to-text, and transmission to Raspberry Pi."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5055,
        groq_api_key: str | None = None,
        timeout: float = 4.0,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.sample_rate = 16_000

        # Initialize speech transcriber
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
            # Show a simple progress counter while recording
            start_t = time.time()
            while sd.wait() is None:
                pass

            pcm_bytes = recording.tobytes()
            print("[RECORDING] Capture complete.")
            return pcm_bytes
        except Exception as exc:
            print(f"[ERROR] Failed recording from laptop microphone: {exc}")
            return None

    def transcribe_audio(self, pcm_bytes: bytes) -> TranscriptionResult:
        """Transcribe PCM bytes using Groq -> faster-whisper -> manual."""
        return self.transcriber.transcribe(pcm_bytes, sample_rate=self.sample_rate)

    def interactive_session(self, default_duration: float = 3.5) -> None:
        """Interactive loop for name enrollment from laptop to Raspberry Pi."""
        print("=" * 64)
        print("       LS-Face Laptop Voice Companion (Input Feeder)       ")
        print("=" * 64)
        print(f"Target Raspberry Pi: {self.base_url}")
        print(f"STT Providers: {self.transcriber.get_initial_status()}")
        print("-" * 64)

        # Check connectivity first
        ok, msg = self.check_connection()
        if ok:
            print(f"[STATUS] OK - {msg}")
        else:
            print(f"[STATUS] WARNING - {msg}")
            print("  Make sure the LS-Face app is open on the Raspberry Pi and on the same Wi-Fi/LAN.")

        print("\nCommands:")
        print("  - Press [Enter] to record your name from laptop mic")
        print("  - Type any name directly (e.g., 'Alice Smith') to transfer immediately")
        print("  - Type 'check' to recheck connection to Raspberry Pi")
        print("  - Type 'q' or 'exit' to quit\n")

        while True:
            try:
                user_cmd = input("Laptop Voice [Enter to speak / type name / 'q']: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

            if user_cmd.lower() in ("q", "exit", "quit"):
                print("Exiting laptop voice companion. Goodbye!")
                break

            if user_cmd.lower() == "check":
                ok, msg = self.check_connection()
                print(f"[CHECK] {'OK' if ok else 'FAIL'} - {msg}")
                continue

            # Option A: User typed a name directly
            if user_cmd:
                print(f"\n[TRANSFER] Sending typed name '{user_cmd}' to Raspberry Pi...")
                trans_ok, trans_msg = self.transfer_name(user_cmd, provider="laptop-manual")
                print(f"[{'SUCCESS' if trans_ok else 'FAILED'}] {trans_msg}\n")
                continue

            # Option B: User pressed Enter to speak
            pcm = self.record_audio(duration_sec=default_duration)
            if not pcm:
                typed = input("Microphone unavailable. Type name manually: ").strip()
                if typed:
                    trans_ok, trans_msg = self.transfer_name(typed, provider="laptop-manual")
                    print(f"[{'SUCCESS' if trans_ok else 'FAILED'}] {trans_msg}\n")
                continue

            # Run transcription on laptop
            print("[TRANSCRIBING] Processing audio with Groq whisper-v3-turbo (primary)...")
            result = self.transcribe_audio(pcm)

            if result.cleaned_name:
                provider_desc = result.provider
                if result.is_fallback:
                    provider_desc += f" (fallback: {result.model})"
                else:
                    provider_desc += f" ({result.model})"

                print(f"[RESULT] Transcribed: '{result.cleaned_name}' via {provider_desc} ({result.latency_sec:.2f}s)")

                # Transfer to Raspberry Pi
                print(f"[TRANSFER] Sending '{result.cleaned_name}' to Raspberry Pi at {self.base_url}...")
                trans_ok, trans_msg = self.transfer_name(
                    result.cleaned_name,
                    provider=result.provider,
                    model=result.model,
                )
                print(f"[{'SUCCESS' if trans_ok else 'FAILED'}] {trans_msg}\n")
            else:
                print(f"[WARN] No speech detected or STT failed: {result.error}")
                typed = input("Would you like to type the name manually? [leave blank to retry]: ").strip()
                if typed:
                    trans_ok, trans_msg = self.transfer_name(typed, provider="laptop-manual")
                    print(f"[{'SUCCESS' if trans_ok else 'FAILED'}] {trans_msg}\n")


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
    return parser.parse_args()


def main():
    args = parse_args()
    companion = LaptopVoiceCompanion(
        host=args.host,
        port=args.port,
        groq_api_key=args.groq_key,
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
