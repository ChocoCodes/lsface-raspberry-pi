"""Keep enrollment focused on the participant selected by pose detection."""
import numpy as np


def participant_crop(frame, bbox, padding=0.2):
    """Return an independent BGR crop with alignment margin around the face."""
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Invalid participant face bounding box.')
    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError('Participant face bounding box is empty.')
    frame_height, frame_width = frame.shape[:2]
    left = max(0, int(np.floor(x - width * padding)))
    top = max(0, int(np.floor(y - height * padding)))
    right = min(frame_width, int(np.ceil(x + width * (1 + padding))))
    bottom = min(frame_height, int(np.ceil(y + height * (1 + padding))))
    if right <= left or bottom <= top:
        raise ValueError('Participant face is outside the frame.')
    return frame[top:bottom, left:right].copy()
