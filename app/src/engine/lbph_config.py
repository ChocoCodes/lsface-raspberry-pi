"""Explicit LBPH descriptor profiles and serialized-model validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LBPHDescriptor:
    descriptor_id: str
    radius: int
    neighbors: int
    grid_x: int
    grid_y: int

    def __post_init__(self) -> None:
        expected = format_descriptor_id(self.radius, self.neighbors, self.grid_x, self.grid_y)
        if self.descriptor_id != expected:
            raise ValueError(f"Descriptor id {self.descriptor_id!r} does not match {expected!r}.")
        if min(self.radius, self.neighbors, self.grid_x, self.grid_y) <= 0:
            raise ValueError("LBPH radius, neighbors, and grid dimensions must be positive.")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "id": self.descriptor_id,
            "radius": self.radius,
            "neighbors": self.neighbors,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
        }


def format_descriptor_id(radius: int, neighbors: int, grid_x: int, grid_y: int) -> str:
    return f"r{radius}_n{neighbors}_g{grid_x}x{grid_y}"


DEPLOYED_DESCRIPTOR = LBPHDescriptor("r1_n8_g8x8", 1, 8, 8, 8)
SELECTED_DESCRIPTOR = LBPHDescriptor("r3_n8_g6x6", 3, 8, 6, 6)


def resolve_descriptor(value: str | LBPHDescriptor = "selected") -> LBPHDescriptor:
    if isinstance(value, LBPHDescriptor):
        return value
    key = str(value).strip().lower()
    choices = {
        "deployed": DEPLOYED_DESCRIPTOR,
        "r1_n8_g8x8": DEPLOYED_DESCRIPTOR,
        "selected": SELECTED_DESCRIPTOR,
        "r3_n8_g6x6": SELECTED_DESCRIPTOR,
    }
    try:
        return choices[key]
    except KeyError as error:
        raise ValueError(f"Unknown LBPH descriptor {value!r}; choose {', '.join(sorted(choices))}.") from error


def descriptor_from_mapping(payload: Mapping[str, Any] | None, *, source: str) -> LBPHDescriptor:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must contain lbph_descriptor metadata.")
    descriptor_id = payload.get("id", payload.get("descriptor_id"))
    required = ("radius", "neighbors", "grid_x", "grid_y")
    missing = [name for name in required if name not in payload]
    if descriptor_id is None or missing:
        detail = []
        if descriptor_id is None:
            detail.append("missing id")
        if missing:
            detail.append("missing " + ", ".join(missing))
        raise ValueError(f"Invalid LBPH metadata in {source}: {'; '.join(detail)}")
    try:
        params = {name: int(payload[name]) for name in required}
        return LBPHDescriptor(str(descriptor_id), **params)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid LBPH metadata in {source}: {error}") from error


def make_lbph(descriptor: str | LBPHDescriptor = SELECTED_DESCRIPTOR):
    import cv2 as cv

    resolved = resolve_descriptor(descriptor)
    return cv.face.LBPHFaceRecognizer_create(
        radius=resolved.radius,
        neighbors=resolved.neighbors,
        grid_x=resolved.grid_x,
        grid_y=resolved.grid_y,
    )


def inspect_lbph(recognizer: Any) -> LBPHDescriptor:
    names = {
        "radius": "getRadius",
        "neighbors": "getNeighbors",
        "grid_x": "getGridX",
        "grid_y": "getGridY",
    }
    missing = [getter for getter in names.values() if not hasattr(recognizer, getter)]
    if missing:
        raise RuntimeError("LBPH recognizer lacks serialized descriptor getters: " + ", ".join(missing))
    values = {key: int(getattr(recognizer, getter)()) for key, getter in names.items()}
    return LBPHDescriptor(format_descriptor_id(**values), **values)


def validate_loaded_lbph(recognizer: Any, expected: str | LBPHDescriptor, *, context: str) -> LBPHDescriptor:
    expected_descriptor = resolve_descriptor(expected)
    observed = inspect_lbph(recognizer)
    if observed != expected_descriptor:
        raise RuntimeError(
            f"LBPH descriptor mismatch for {context}: expected {expected_descriptor.descriptor_id}, "
            f"observed {observed.descriptor_id}. Use matching model, release, and thresholds."
        )
    return observed


def release_descriptor_from_manifest(
    manifest: Mapping[str, Any], expected: str | LBPHDescriptor, *, source: str
) -> LBPHDescriptor:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{source} must be a JSON object.")
    descriptor = descriptor_from_mapping(manifest.get("lbph_descriptor"), source=source)
    manifest_id = manifest.get("descriptor_id")
    if manifest_id != descriptor.descriptor_id:
        raise ValueError(f"{source} descriptor_id disagrees with lbph_descriptor.id.")
    expected_descriptor = resolve_descriptor(expected)
    if descriptor != expected_descriptor:
        raise ValueError(
            f"{source} expects {descriptor.descriptor_id}; thresholds expect {expected_descriptor.descriptor_id}."
        )
    return descriptor
