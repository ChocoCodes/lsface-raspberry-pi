"""Distinguish short UI stalls from repeated camera frames."""


class PreviewTiming:
    def __init__(self, logger, threshold=0.2):
        self.logger = logger
        self.threshold = threshold
        self._last_tick = None
        self._frame = None
        self._fresh_at = None
        self._reported = False

    def observe(self, frame, now):
        if self._last_tick is not None and now - self._last_tick >= self.threshold:
            self.logger.warning('[PreviewTiming] UI callback gap: %.0f ms', (now - self._last_tick) * 1000)
        self._last_tick = now
        if frame is not self._frame:
            if self._reported:
                self.logger.warning('[PreviewTiming] New camera frame after %.0f ms', (now - self._fresh_at) * 1000)
            self._frame = frame
            self._fresh_at = now
            self._reported = False
        elif frame is not None and self._fresh_at is not None and not self._reported:
            if now - self._fresh_at >= self.threshold:
                self.logger.warning('[PreviewTiming] Camera still returning the same frame after %.0f ms', (now - self._fresh_at) * 1000)
                self._reported = True
