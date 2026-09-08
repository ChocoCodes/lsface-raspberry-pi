from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
from uuid import UUID, uuid4

import numpy as np

from src.config.config import (
    CUSTOM_DATABASE_ROOT,
    DATABASE_CATALOG_PATH,
    DB,
    ENROLLMENT_ROOT,
)
from src.engine.rebuild_release import build_release, load_records
from src.engine.lbph_config import resolve_descriptor

from .feature_db import FeatureDB


@dataclass(frozen=True)
class DatabaseSpec:
    id: str
    name: str
    db_path: Path
    enrollment_root: Path


@dataclass(frozen=True)
class PreparedEnrollment:
    """Feature samples prepared before the operator confirms an identity name."""

    samples: tuple[tuple[np.ndarray, np.ndarray], ...]
    database_id: str | None = None


CATALOG_PATH = Path(DATABASE_CATALOG_PATH)
CUSTOM_DATABASES_PATH = Path(CUSTOM_DATABASE_ROOT)


class DatabaseManager:
    """Coordinate editable database snapshots and live-release artifacts."""

    _lock = threading.RLock()
    _default_database = next(iter(DB.values()))
    _builtin_id = "lasalle"
    _builtin_name = "La Salle Database"

    @classmethod
    def _default_catalog(cls) -> dict:
        return {
            "schema_version": 1,
            "selected_id": cls._builtin_id,
            "databases": [
                {"id": cls._builtin_id, "name": cls._builtin_name, "kind": "builtin"}
            ],
        }

    @classmethod
    def _write_catalog(cls, catalog: dict) -> None:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=CATALOG_PATH.parent,
            prefix=".catalog-",
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(catalog, handle, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, CATALOG_PATH)

    @classmethod
    def _read_catalog(cls) -> dict:
        if not CATALOG_PATH.is_file():
            catalog = cls._default_catalog()
            cls._write_catalog(catalog)
            return catalog

        try:
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read database catalog: {CATALOG_PATH}") from exc
        if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported database catalog: {CATALOG_PATH}")

        entries = catalog.get("databases")
        if not isinstance(entries, list):
            raise RuntimeError(f"Database catalog has no database list: {CATALOG_PATH}")

        ids: set[str] = set()
        names: set[str] = set()
        normalized_entries = []
        changed = False
        for source_entry in entries:
            if not isinstance(source_entry, dict):
                raise RuntimeError(f"Invalid database catalog entry: {source_entry!r}")
            entry = dict(source_entry)
            database_id = entry.get("id")
            if not isinstance(database_id, str) or not database_id or database_id in ids:
                raise RuntimeError(f"Invalid or duplicate database id: {database_id!r}")
            if database_id != cls._builtin_id:
                try:
                    UUID(database_id)
                except (ValueError, AttributeError) as exc:
                    raise RuntimeError(f"Custom database id is not a UUID: {database_id!r}") from exc
                if not isinstance(entry.get("directory"), str) or not entry["directory"]:
                    raise RuntimeError(f"Custom database {database_id!r} has no directory.")
                try:
                    normalized_name = FeatureDB.normalize_name(entry.get("name"))
                except ValueError as exc:
                    raise RuntimeError(f"Invalid name for database {database_id!r}") from exc
            else:
                normalized_name = cls._builtin_name

            name_key = normalized_name.casefold()
            if name_key in names:
                raise RuntimeError(f"Duplicate database name: {normalized_name!r}")
            if entry.get("name") != normalized_name:
                entry["name"] = normalized_name
                changed = True
            ids.add(database_id)
            names.add(name_key)
            normalized_entries.append(entry)

        if cls._builtin_id not in ids:
            if cls._builtin_name.casefold() in names:
                raise RuntimeError("The reserved La Salle database name is already in use.")
            normalized_entries.insert(0, cls._default_catalog()["databases"][0])
            ids.add(cls._builtin_id)
            catalog["selected_id"] = cls._builtin_id
            changed = True
        elif catalog.get("selected_id") not in ids:
            catalog["selected_id"] = cls._builtin_id
            changed = True

        if changed or catalog.get("databases") != normalized_entries:
            catalog["databases"] = normalized_entries
            cls._write_catalog(catalog)
        return catalog

    @classmethod
    def _spec_from_entry(cls, entry: dict) -> DatabaseSpec:
        database_id = entry["id"]
        if database_id == cls._builtin_id:
            return DatabaseSpec(
                id=database_id,
                name=cls._builtin_name,
                db_path=Path(DB[cls._builtin_name]).resolve(),
                enrollment_root=Path(ENROLLMENT_ROOT).resolve(),
            )

        relative_directory = entry.get("directory")
        if not isinstance(relative_directory, str) or not relative_directory:
            raise RuntimeError(f"Custom database {database_id!r} has no directory.")
        catalog_root = CATALOG_PATH.parent.resolve()
        enrollment_root = (catalog_root / relative_directory).resolve()
        custom_root = CUSTOM_DATABASES_PATH.resolve()
        if catalog_root not in enrollment_root.parents or custom_root not in enrollment_root.parents:
            raise RuntimeError(f"Database {database_id!r} escapes the local database directory.")
        return DatabaseSpec(
            id=database_id,
            name=FeatureDB.normalize_name(entry["name"]),
            db_path=enrollment_root / "database.npy",
            enrollment_root=enrollment_root,
        )

    @classmethod
    def list_databases(cls) -> list[DatabaseSpec]:
        with cls._lock:
            catalog = cls._read_catalog()
            return [cls._spec_from_entry(entry) for entry in catalog["databases"]]

    @classmethod
    def selected_database_id(cls) -> str:
        with cls._lock:
            return str(cls._read_catalog()["selected_id"])

    @classmethod
    def resolve_database(cls, database_id: str) -> DatabaseSpec:
        with cls._lock:
            catalog = cls._read_catalog()
            for entry in catalog["databases"]:
                if entry.get("id") == database_id:
                    return cls._spec_from_entry(entry)
        raise KeyError(f"Unknown database id: {database_id!r}")

    @classmethod
    def select_database(cls, database_id: str) -> DatabaseSpec:
        with cls._lock:
            catalog = cls._read_catalog()
            selected = None
            for entry in catalog["databases"]:
                if entry.get("id") == database_id:
                    selected = cls._spec_from_entry(entry)
                    break
            if selected is None:
                raise KeyError(f"Unknown database id: {database_id!r}")
            catalog["selected_id"] = database_id
            cls._write_catalog(catalog)
            return selected

    @classmethod
    def create_database(cls, name: str) -> DatabaseSpec:
        """Create and select an empty database with an isolated release root."""

        normalized_name = FeatureDB.normalize_name(name)
        with cls._lock:
            catalog = cls._read_catalog()
            if any(
                FeatureDB.normalize_name(entry["name"]).casefold() == normalized_name.casefold()
                for entry in catalog["databases"]
            ):
                raise ValueError(f"Database {normalized_name!r} already exists.")

            database_id = str(uuid4())
            catalog_root = CATALOG_PATH.parent.resolve()
            custom_root = CUSTOM_DATABASES_PATH.resolve()
            if catalog_root not in custom_root.parents:
                raise RuntimeError("Custom database storage must remain inside the catalog directory.")
            relative_directory = custom_root.relative_to(catalog_root) / database_id
            enrollment_root = custom_root / database_id
            enrollment_root.mkdir(parents=True, exist_ok=False)
            database_path = enrollment_root / "database.npy"
            cls._write_database_payload(database_path, {})
            # Publish the empty release before exposing the entry in the catalog.
            cls._publish_payload({}, database_path, enrollment_root)

            catalog["databases"].append(
                {
                    "id": database_id,
                    "name": normalized_name,
                    "kind": "custom",
                    "directory": relative_directory.as_posix(),
                }
            )
            catalog["selected_id"] = database_id
            cls._write_catalog(catalog)
            return DatabaseSpec(
                id=database_id,
                name=normalized_name,
                db_path=database_path,
                enrollment_root=enrollment_root,
            )

    @classmethod
    def _write_database_payload(cls, path: Path, database: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".database-",
            suffix=".npy",
            delete=False,
        ) as handle:
            np.save(handle, database, allow_pickle=True)
            temporary = Path(handle.name)
        os.replace(temporary, path)

    @classmethod
    def _paths(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> tuple[Path, Path]:
        if database_id is not None:
            if db_path is not None or enrollment_root is not None:
                raise ValueError("database_id cannot be combined with db_path or enrollment_root")
            spec = cls.resolve_database(database_id)
            return spec.db_path, spec.enrollment_root

        # Existing callers that omit paths follow the catalog selection. Calls
        # that provide either path keep the original path-based behavior.
        if db_path is None and enrollment_root is None:
            selected = cls.resolve_database(cls.selected_database_id())
            return selected.db_path, selected.enrollment_root
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
            if enrollment_root.resolve() != Path(ENROLLMENT_ROOT).resolve():
                raise RuntimeError(f"Active release has no database snapshot: {release_root}")
            return None
        return snapshot

    @classmethod
    def active_database_path(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> Path:
        database, root = cls._paths(db_path, enrollment_root, database_id=database_id)
        return cls._active_snapshot_path(root) or database

    @classmethod
    def load(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> FeatureDB:
        """Load the active editable snapshot, falling back to the base DB."""

        with cls._lock:
            return FeatureDB.load(
                cls.active_database_path(db_path, enrollment_root, database_id=database_id)
            )

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

        cls._publish_payload(candidate.db, source_database, enrollment_root)
        return candidate

    @classmethod
    def _publish_payload(
        cls,
        database: dict,
        source_database: Path,
        enrollment_root: Path,
    ) -> None:
        """Build a release from a payload without loading the feature models."""

        FeatureDB.validate_database(database)
        temporary_database = cls._temporary_database(database, enrollment_root)
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

    @classmethod
    def ensure_release(
        cls,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> FeatureDB:
        """Create the first live release when this database has none yet."""

        database, root = cls._paths(db_path, enrollment_root, database_id=database_id)
        with cls._lock:
            active = cls._active_snapshot_path(root)
            if active is not None:
                return FeatureDB.load(active)
            base = FeatureDB.load(database)
            return cls._publish(base, database, root)

    @classmethod
    def _frame_values(
        cls,
        frames: Iterable[np.ndarray] | Mapping[str, np.ndarray] | np.ndarray,
    ) -> list[np.ndarray]:
        if isinstance(frames, np.ndarray):
            return [frames]
        if isinstance(frames, Mapping):
            return list(frames.values())
        return list(frames)

    @staticmethod
    def _identity_name(database: dict, name: str) -> str | None:
        name_key = name.casefold()
        return next(
            (existing for existing in database if existing.casefold() == name_key),
            None,
        )

    @classmethod
    def prepare_enrollment(
        cls,
        frames: Iterable[np.ndarray] | Mapping[str, np.ndarray] | np.ndarray,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> PreparedEnrollment:
        """Extract samples without changing the selected database.

        The returned value can be passed to :meth:`commit_enrollment` after a
        name is collected. Extraction uses a discarded ``FeatureDB`` clone so
        the live database and its release pointer remain untouched.
        """

        frame_values = cls._frame_values(frames)
        if not frame_values:
            raise ValueError("At least one enrollment frame is required.")

        with cls._lock:
            prepared_database_id = database_id
            if prepared_database_id is None and db_path is None and enrollment_root is None:
                prepared_database_id = cls.selected_database_id()
            database, root = cls._paths(
                db_path,
                enrollment_root,
                database_id=prepared_database_id,
            )
            current = FeatureDB.load(cls._active_snapshot_path(root) or database)

        working = current.clone()
        temporary_name = f"__prepared_{uuid4().hex}"
        for frame in frame_values:
            # FeatureDB.enroll_frame is the existing extraction primitive. The
            # temporary identity lives only on this discarded clone.
            working.enroll_frame(temporary_name, frame)

        record = working.db[temporary_name]
        samples = tuple(
            (
                np.asarray(lbph, dtype=np.uint8).copy(),
                np.asarray(sface, dtype=np.float32).reshape(-1).copy(),
            )
            for lbph, sface in zip(record["lbph"], record["sface"])
        )
        return PreparedEnrollment(samples=samples, database_id=prepared_database_id)

    @classmethod
    def commit_enrollment(
        cls,
        name: str,
        prepared_features: PreparedEnrollment | Iterable[tuple[np.ndarray, np.ndarray]],
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> FeatureDB:
        """Commit prepared samples against the latest database snapshot."""

        normalized_name = FeatureDB.normalize_name(name)
        prepared_id = (
            prepared_features.database_id
            if isinstance(prepared_features, PreparedEnrollment)
            else None
        )
        if prepared_id is not None:
            if database_id is not None and database_id != prepared_id:
                raise ValueError("Prepared enrollment belongs to a different database.")
            database_id = prepared_id

        samples = (
            prepared_features.samples
            if isinstance(prepared_features, PreparedEnrollment)
            else tuple(
                prepared_features.values()
                if isinstance(prepared_features, Mapping)
                else prepared_features
            )
        )
        if not samples:
            raise ValueError("At least one prepared enrollment sample is required.")

        database, root = cls._paths(db_path, enrollment_root, database_id=database_id)
        with cls._lock:
            # Re-read after the name has been confirmed so concurrent changes
            # cannot be overwritten by an older preparation snapshot.
            current = FeatureDB.load(cls._active_snapshot_path(root) or database)
            existing_name = cls._identity_name(current.db, normalized_name)
            if existing_name is not None:
                raise ValueError(f"Identity {existing_name!r} already exists.")

            candidate = current.clone()
            identity_id = max(
                (int(record["id"]) for record in candidate.db.values()), default=-1
            ) + 1
            record = {"id": identity_id, "lbph": [], "sface": []}
            for sample in samples:
                if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                    raise ValueError("Prepared enrollment samples must contain LBPH/SFace pairs.")
                lbph, sface = sample
                lbph_array = np.asarray(lbph, dtype=np.uint8).copy()
                sface_array = np.asarray(sface, dtype=np.float32).reshape(-1).copy()
                if lbph_array.shape != (100, 100) or sface_array.size != 128:
                    raise ValueError("Prepared enrollment contains an invalid feature shape.")
                if not np.isfinite(sface_array).all():
                    raise ValueError("Prepared enrollment contains a non-finite SFace feature.")
                record["lbph"].append(lbph_array)
                record["sface"].append(sface_array)
            candidate.db[normalized_name] = record
            return cls._publish(candidate, database, root)

    @classmethod
    def enroll(
        cls,
        name: str,
        frames: Iterable[np.ndarray] | Mapping[str, np.ndarray] | np.ndarray,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> FeatureDB:
        """Backward-compatible wrapper for prepare + commit enrollment."""

        prepared = cls.prepare_enrollment(
            frames,
            db_path=db_path,
            enrollment_root=enrollment_root,
            database_id=database_id,
        )
        return cls.commit_enrollment(
            name,
            prepared,
            db_path=db_path,
            enrollment_root=enrollment_root,
            database_id=database_id,
        )

    @classmethod
    def delete(
        cls,
        name: str,
        db_path: str | Path | None = None,
        enrollment_root: str | Path | None = None,
        *,
        database_id: str | None = None,
    ) -> FeatureDB:
        """Delete one identity and publish the rebuilt live release."""

        normalized_name = FeatureDB.normalize_name(name)
        database, root = cls._paths(db_path, enrollment_root, database_id=database_id)
        with cls._lock:
            current = FeatureDB.load(cls._active_snapshot_path(root) or database)
            existing_name = cls._identity_name(current.db, normalized_name)
            if existing_name is None:
                raise KeyError(f"Identity {normalized_name!r} does not exist.")

            candidate = current.clone()
            del candidate.db[existing_name]
            return cls._publish(candidate, database, root)
