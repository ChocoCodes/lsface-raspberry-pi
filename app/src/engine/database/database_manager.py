from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src.config.config import DB, ENROLLMENT_ROOT
from src.engine.rebuild_release import build_release, load_records
from src.engine.lbph_config import resolve_descriptor

from .feature_db import FeatureDB


class DatabaseManager:
    """Coordinate editable database snapshots and live-release artifacts."""

    _lock = threading.RLock()
    _default_database = next(iter(DB.values()))

    @classmethod
    def _paths(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> tuple[Path, Path]:
        database = Path(db_path) if db_path is not None else Path(cls._default_database)
        root = Path(enrollment_root) if enrollment_root is not None else Path(ENROLLMENT_ROOT)
        return database.resolve(), root.resolve()

    @classmethod
    def _active_snapshot_path(cls, enrollment_root: Path) -> Path | None:
        pointer = enrollment_root / "current.json"
        if not pointer.is_file():
            return None

        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            release_name = payload["release"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Invalid active enrollment pointer: {pointer}") from exc

        release_root = (enrollment_root / str(release_name)).resolve()
        if enrollment_root not in release_root.parents:
            raise RuntimeError("Active enrollment pointer escapes the enrollment directory.")

        snapshot_name = "database.npy"
        manifest = release_root / "manifest.json"
        if manifest.is_file():
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                snapshot_name = str(manifest_payload.get("database_snapshot", snapshot_name))
            except (OSError, ValueError, TypeError) as exc:
                raise RuntimeError(f"Invalid enrollment manifest: {manifest}") from exc

        snapshot = (release_root / snapshot_name).resolve()
        if release_root not in snapshot.parents or not snapshot.is_file():
            return None
        return snapshot

    @classmethod
    def active_database_path(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> Path:
        database, root = cls._paths(db_path, enrollment_root)
        return cls._active_snapshot_path(root) or database

    @classmethod
    def load(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> FeatureDB:
        """Load the active editable snapshot, falling back to the base DB."""

        with cls._lock:
            return FeatureDB.load(cls.active_database_path(db_path, enrollment_root))

    @classmethod
    def _temporary_database(cls, database: dict, enrollment_root: Path) -> Path:
        enrollment_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=enrollment_root,
            prefix=".database-",
            suffix=".npy",
            delete=False,
        ) as handle:
            np.save(handle, database, allow_pickle=True)
            return Path(handle.name)

    @classmethod
    def _publish(
        cls,
        candidate: FeatureDB,
        source_database: Path,
        enrollment_root: Path,
    ) -> FeatureDB:
        """Build a candidate release and atomically advance ``current.json``."""

        FeatureDB.validate_database(candidate.db)
        temporary_database = cls._temporary_database(candidate.db, enrollment_root)
        try:
            records, selection = load_records(
                temporary_database,
                samples_per_identity=10,
                selection_seed=42,
                allow_fewer=True,
            )
            build_release(
                records,
                temporary_database,
                enrollment_root,
                resolve_descriptor("r3_n8_g6x6"),
                selection,
                source_database=source_database,
            )
        finally:
            try:
                temporary_database.unlink()
            except FileNotFoundError:
                pass
        return candidate

    @classmethod
    def ensure_release(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> FeatureDB:
        """Create the first live release when this database has none yet."""

        database, root = cls._paths(db_path, enrollment_root)
        with cls._lock:
            active = cls._active_snapshot_path(root)
            if active is not None:
                return FeatureDB.load(active)
            base = FeatureDB.load(database)
            return cls._publish(base, database, root)

    @classmethod
    def enroll(
        cls,
        name: str,
        frames: Iterable[np.ndarray] | Mapping[str, np.ndarray],
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> FeatureDB:
        """Enroll all supplied frames and publish one atomic live release."""

        normalized_name = FeatureDB.normalize_name(name)
        frame_values = list(frames.values()) if isinstance(frames, Mapping) else list(frames)
        if not frame_values:
            raise ValueError("At least one enrollment frame is required.")

        database, root = cls._paths(db_path, enrollment_root)
        with cls._lock:
            current = FeatureDB.load(cls._active_snapshot_path(root) or database)
            if normalized_name in current.db:
                raise ValueError(f"Identity {normalized_name!r} already exists.")

            candidate = current.clone()
            for frame in frame_values:
                candidate.enroll_frame(normalized_name, frame)
            return cls._publish(candidate, database, root)

    @classmethod
    def delete(
        cls,
        name: str,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
    ) -> FeatureDB:
        """Delete one identity and publish the rebuilt live release."""

        normalized_name = FeatureDB.normalize_name(name)
        database, root = cls._paths(db_path, enrollment_root)
        with cls._lock:
            current = FeatureDB.load(cls._active_snapshot_path(root) or database)
            if normalized_name not in current.db:
                raise KeyError(f"Identity {normalized_name!r} does not exist.")

            candidate = current.clone()
            del candidate.db[normalized_name]
            return cls._publish(candidate, database, root)
