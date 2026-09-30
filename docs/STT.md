# PC speech input for existing face enrollment

Run the normal Kivy application on the Pi (`python app/main.py`, adding
`--picamera2` when appropriate). Complete the existing pose capture and leave
the name-confirmation screen open. That screen starts its voice endpoint on
port 5055. No separate receiver or headless camera runner is needed.

On the PC, run:

```powershell
python scripts/laptop_voice_companion.py --host <pi-ip> --sttloc pi
```

Press Enter to record, or type a name. Preview the proposed name, edit it if
needed, then accept. Acceptance is dispatched onto Kivy's UI thread and invokes
the screen's existing Continue/save flow. The companion reports completion
only after the application reports that enrollment was saved. Cancel returns
the application to Home and discards temporary speech results.

Set `GROQ_API_KEY` in the **Pi application's environment**, before starting the
app. Merely creating `.env` does not load it. Groq is primary; the existing
faster-whisper adapter provides local fallback. Provider errors also allow
typed input. Install `sounddevice` and `numpy` on the PC; the Pi needs the
application dependencies plus `groq` and optionally `faster-whisper`.

`LSFACE_REMOTE_VOICE_PORT` changes the application port; pass the matching
`--port` to the companion. A busy port produces an error on the name screen
instead of silently switching ports. Use a trusted LAN; the endpoint has no
authentication or TLS.

The STT service opens neither a camera nor a Pi microphone. The existing app
still uses its camera for pose capture and live recognition and still requires
Kivy's desktop environment. Do not start the full app on the PC unless you
intend to use its camera for normal enrollment.

Only one current enrollment and its latest proposed name are held in memory.
Old session IDs, late transcriptions, and replaced recordings are rejected.
On completion, failure, or cancellation, name proposals are discarded; only
the current session ID and state remain available for status polling. The
existing face database is written by the existing enrollment code, never by
the voice server. In-progress database writes cannot be cancelled.

The companion's Pi mode is interactive; `--once`, `--audio-file`, and `--text`
are rejected. Type a name at the interactive prompt for manual fallback.
Legacy `--sttloc pc` targets the older remote-name receiver, not this protocol.

Automated checks (no camera, cloud requests, or enrollment writes):

```powershell
python -m unittest discover -s test -p test_enrollment_voice.py -v
```
