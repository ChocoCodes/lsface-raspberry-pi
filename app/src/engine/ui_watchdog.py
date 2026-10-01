"""Capture Python thread stacks when Kivy stops servicing its clock."""

import faulthandler
from datetime import datetime
from pathlib import Path


class UIWatchdog:
    def __init__(self, clock, root, path, timeout=3.0):
        self.clock = clock
        self.root = root
        self.path = Path(path)
        self.timeout = timeout
        self._file = None
        self._event = None
        self._state = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('a', encoding='utf-8', buffering=1)
        self._file.write(f'\nSession started {datetime.now().isoformat()}\n')
        self._heartbeat(0)
        self._event = self.clock.schedule_interval(self._heartbeat, 0.5)

    def _heartbeat(self, _dt):
        screen = self.root.current_screen
        state = (self.root.current, getattr(screen, 'phase', ''), getattr(screen, 'mode', ''))
        if state != self._state:
            self._file.write(f'{datetime.now().isoformat()} screen/phase/mode={state!r}\n')
            self._state = state
        # Native watchdog still dumps stacks when Python's UI thread is blocked.
        faulthandler.dump_traceback_later(self.timeout, repeat=True, file=self._file)

    def close(self):
        if self._event is not None:
            self._event.cancel()
            self._event = None
        faulthandler.cancel_dump_traceback_later()
        if self._file is not None:
            self._file.close()
            self._file = None
