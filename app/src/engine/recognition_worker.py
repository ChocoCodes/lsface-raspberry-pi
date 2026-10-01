"""Single-flight inference; only the caller thread consumes results."""

from concurrent.futures import ThreadPoolExecutor
import time


class RecognitionWorker:
    def __init__(self):
        self._executor = None
        self._future = None
        self._generation = 0
        self._job_generation = 0

    def submit(self, cascade, frame, captured_at):
        if self._future is not None:
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recognition")
        self._job_generation = self._generation
        self._future = self._executor.submit(self._infer, cascade, frame.copy(), captured_at)
        return True

    @staticmethod
    def _infer(cascade, frame, captured_at):
        started = time.monotonic()
        try:
            results = cascade.infer(frame)
            error = None
        except Exception as exc:
            results, error = None, exc
        return captured_at, frame.shape, results, (time.monotonic() - started) * 1000, error

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        future, self._future = self._future, None
        if self._job_generation != self._generation:
            return None
        return future.result()

    def close(self):
        self._generation += 1
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        # Retain the in-flight job until it finishes, preventing overlap on re-entry.
        # It owns its cascade and snapshot and cannot touch the screen or camera.
