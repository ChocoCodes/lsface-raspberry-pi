"""Bounded registration diagnostics; no images or face embeddings are logged."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading

_context = ContextVar('registration_context', default={})
_lock = threading.Lock()


class _ContextFilter(logging.Filter):
    def filter(self, record):
        context = _context.get()
        record.session = context.get('session', '-')
        record.pose = context.get('pose', '-')
        return True


def registration_logger():
    logger = logging.getLogger('lsface.registration')
    with _lock:
        if not logger.handlers:
            path = Path(__file__).resolve().parents[2] / 'logs' / 'registration.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding='utf-8')
            handler.addFilter(_ContextFilter())
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s session=%(session)s pose=%(pose)s %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
    return logger


@contextmanager
def registration_context(**values):
    token = _context.set({**_context.get(), **values})
    try:
        yield
    finally:
        _context.reset(token)
