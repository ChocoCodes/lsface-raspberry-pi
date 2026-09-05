#!/usr/bin/env python3
"""Build paired LBPH/SFace releases from the upstream enrollment cohort.

For the pulled upstream ``lasalledb.npy``, defaults reproduce its shipped
LBPH YAML exactly: 10 rows per identity, selected in identity-id order with
``numpy.random.default_rng(42).choice(..., replace=False)``. The source DB is
read-only; only this integration's selected release directory is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from lbph_config import LBPHDescriptor, make_lbph, resolve_descriptor


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT
DEFAULT_DATABASE = ROOT / "db" / "lasalledb.npy"
DEFAULT_OUTPUT_ROOT = ROOT / "enrollment"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--descriptor",
        default="selected",
        choices=("deployed", "r1_n8_g8x8", "selected", "r3_n8_g6x6"),
    )
    parser.add_argument(
        "--samples-per-identity",
        type=int,
        default=10,
        help="Upstream .npy cohort size per identity; 10 reproduces the upstream YAML.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Seed for numpy.default_rng(...).choice(..., replace=False).",
    )
    parser.add_argument(
        "--release-name",
        help=(
            "Stable directory name under releases/. If omitted, create a unique "
            "timestamped build for local experiments."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_record(label: str, tile: object, feature: object) -> tuple[np.ndarray, np.ndarray]:
    tile_array = np.asarray(tile, dtype=np.uint8)
    feature_array = np.asarray(feature, dtype=np.float32).reshape(-1)
    if tile_array.shape != (100, 100):
        raise ValueError(f"{label!r} has LBPH tile shape {tile_array.shape}; expected (100, 100).")
    if feature_array.size != 128 or not np.isfinite(feature_array).all():
        raise ValueError(f"{label!r} has invalid SFace feature shape/content.")
    return tile_array, feature_array


def load_records(
    database: Path, *, samples_per_identity: int = 10, selection_seed: int = 42
) -> tuple[dict[str, list], dict]:
    """Load records and return ``(records, selection_provenance)``."""

    if not database.exists():
        raise FileNotFoundError(database)

    if database.suffix.lower() == ".npy":
        payload = np.load(database, allow_pickle=True).item()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict in {database}.")
        if samples_per_identity < 0:
            raise ValueError("samples_per_identity must be >= 0.")

        records = {
            "labels": [],
            "identity_ids": [],
            "lbph_faces": [],
            "sface_embeddings": [],
        }
        selection = {
            "policy": "numpy.default_rng(seed).choice(per identity, replace=False)",
            "seed": int(selection_seed),
            "samples_per_identity": int(samples_per_identity),
            "identity_order": "record['id'] ascending",
            "indices_by_identity": {},
        }
        rng = np.random.default_rng(selection_seed)
        ordered = sorted(payload.items(), key=lambda item: int(item[1]["id"]))
        for label, person in ordered:
            if not isinstance(person, dict) or "lbph" not in person or "sface" not in person:
                raise ValueError(f"Upstream record {label!r} lacks lbph/sface arrays.")
            if len(person["lbph"]) != len(person["sface"]):
                raise ValueError(f"Upstream record {label!r} has mismatched LBPH/SFace counts.")

            available = len(person["lbph"])
            if samples_per_identity == 0:
                indices = np.arange(available, dtype=np.int64)
            else:
                if samples_per_identity > available:
                    raise ValueError(
                        f"Identity {label!r} has {available} rows; cannot select "
                        f"{samples_per_identity}."
                    )
                indices = rng.choice(available, samples_per_identity, replace=False)
            selection["indices_by_identity"][str(label)] = [int(index) for index in indices]

            for index in indices:
                tile = person["lbph"][int(index)]
                feature = person["sface"][int(index)]
                tile_array, feature_array = _validate_record(str(label), tile, feature)
                records["labels"].append(str(label))
                records["identity_ids"].append(int(person["id"]))
                records["lbph_faces"].append(tile_array)
                records["sface_embeddings"].append(feature_array)
        return records, selection

    with np.load(database, allow_pickle=False) as raw:
        required = {"labels", "lbph_faces", "sface_embeddings"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"{database} missing arrays: {sorted(missing)}")
        labels = raw["labels"].tolist()
        tiles = raw["lbph_faces"].tolist()
        features = raw["sface_embeddings"].tolist()
    if not (len(labels) == len(tiles) == len(features)):
        raise ValueError(f"{database} arrays have different lengths.")

    identity_ids = {name: index for index, name in enumerate(sorted(set(map(str, labels))))}
    records = {
        "labels": [],
        "identity_ids": [],
        "lbph_faces": [],
        "sface_embeddings": [],
    }
    for label, tile, feature in zip(labels, tiles, features):
        tile_array, feature_array = _validate_record(str(label), tile, feature)
        records["labels"].append(str(label))
        records["identity_ids"].append(identity_ids[str(label)])
        records["lbph_faces"].append(tile_array)
        records["sface_embeddings"].append(feature_array)
    return records, {
        "policy": "workspace .npz rows as stored",
        "seed": None,
        "samples_per_identity": None,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _portable_source_path(database: Path) -> str:
    """Record a checkout-relative source path instead of a machine path."""

    resolved = database.resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _release_directory_name(release_name: str | None) -> str:
    if release_name is None:
        return f"release-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    candidate = Path(release_name)
    if (
        not release_name
        or candidate.is_absolute()
        or candidate.name != release_name
        or release_name in {".", ".."}
    ):
        raise ValueError("release name must be a single non-empty directory name")
    return release_name


def build_release(
    records: dict[str, list],
    database: Path,
    output_root: Path,
    descriptor: LBPHDescriptor,
    selection: dict,
    release_name: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    release_root = output_root / "releases"
    release_root.mkdir(parents=True, exist_ok=True)
    final = release_root / _release_directory_name(release_name)
    staging = Path(tempfile.mkdtemp(dir=release_root, prefix=".pending-"))

    identity_ids = {
        name: int(identity_id)
        for name, identity_id in zip(records["labels"], records["identity_ids"])
    }
    names = sorted(identity_ids, key=lambda name: identity_ids[name])
    labels_map = {name: identity_ids[name] for name in names}
    status = "candidate_only" if descriptor.descriptor_id == "r3_n8_g6x6" else "paired_baseline"

    try:
        lbph = make_lbph(descriptor)
        labels = np.asarray(records["identity_ids"], dtype=np.int32)
        faces = [np.asarray(tile, dtype=np.uint8) for tile in records["lbph_faces"]]
        lbph.train(faces, labels)
        lbph.save(str(staging / "lbph.yml"))
        _write_json(staging / "labels.json", labels_map)

        features = np.vstack(records["sface_embeddings"]).astype(np.float32)
        label_array = np.asarray(records["labels"], dtype=str)
        gallery = {
            name: features[label_array == name].mean(axis=0, keepdims=True).astype(np.float32)
            for name in names
        }
        np.save(staging / "sface_gallery.npy", gallery, allow_pickle=True)
        _write_json(staging / "manifest.json", {
            "created_utc": now(),
            "status": status,
            "source_database": _portable_source_path(database),
            "source_format": database.suffix.lower().lstrip("."),
            "identities": names,
            "samples": len(records["labels"]),
            "descriptor_id": descriptor.descriptor_id,
            "lbph_descriptor": descriptor.to_dict(),
            "recipe": "Same selected upstream cohort for LBPH and SFace; explicit LBPH constructor",
            "selection": selection,
        })
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    pointer = output_root / "current.json"
    with tempfile.NamedTemporaryFile(
        dir=output_root, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as handle:
        json.dump({"release": final.relative_to(output_root).as_posix(), "updated_utc": now()}, handle, indent=2)
        temporary_pointer = Path(handle.name)
    os.replace(temporary_pointer, pointer)
    return final


def main() -> int:
    options = parse_args()
    database = options.database if options.database.is_absolute() else (Path.cwd() / options.database).resolve()
    output_root = options.output_root if options.output_root.is_absolute() else (ROOT / options.output_root).resolve()
    descriptor = resolve_descriptor(options.descriptor)
    records, selection = load_records(
        database,
        samples_per_identity=options.samples_per_identity,
        selection_seed=options.selection_seed,
    )
    counts = Counter(records["labels"])
    if len(counts) < 2:
        raise RuntimeError("Need at least two identities.")
    if options.dry_run:
        print(json.dumps({
            "database": _portable_source_path(database),
            "identities": len(counts),
            "samples": len(records["labels"]),
            "descriptor": descriptor.to_dict(),
            "status": "candidate_only" if descriptor.descriptor_id == "r3_n8_g6x6" else "paired_baseline",
            "selection": selection,
        }, indent=2))
        return 0
    release = build_release(
        records,
        database,
        output_root,
        descriptor,
        selection,
        release_name=options.release_name,
    )
    print(f"[RELEASE] descriptor={descriptor.descriptor_id}")
    print(f"[RELEASE] identities={len(counts)} samples={len(records['labels'])}")
    print(f"[RELEASE] selection={selection['policy']} seed={selection['seed']} per_identity={selection['samples_per_identity']}")
    print(f"[RELEASE] active={release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
