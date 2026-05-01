"""Adapter for the ImageCAS coronary CTA dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VOLUME_EXTENSIONS = (".nii", ".nii.gz", ".mhd", ".mha", ".nrrd", ".dcm")
MESH_EXTENSIONS = (".stl", ".obj", ".ply", ".off", ".dae")


@dataclass(frozen=True)
class ImageCASVolumeCase:
    """One ImageCAS CTA case, optionally paired with a vessel segmentation or mesh."""

    case_id: str
    volume_path: Path | None = None
    mesh_path: Path | None = None
    segmentation_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_mesh(self) -> bool:
        return self.mesh_path is not None


@dataclass(frozen=True)
class ImageCASCases:
    """Container for discovered ImageCAS cases."""

    volume_cases: list[ImageCASVolumeCase]
    mesh_cases: list[ImageCASVolumeCase]


class ImageCASAdapter:
    """Discover ImageCAS CTA volumes and any derived meshes."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def discover(self) -> ImageCASCases:
        volume_cases: list[ImageCASVolumeCase] = []
        mesh_cases: list[ImageCASVolumeCase] = []

        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            case_id = path.stem
            if suffix == ".gz" and path.name.endswith(".nii.gz"):
                case_id = path.name[:-7]
            if case_id.endswith("_mesh"):
                case_id = case_id[:-5]

            if path.name.endswith(".nii.gz") or suffix in VOLUME_EXTENSIONS:
                volume_cases.append(
                    ImageCASVolumeCase(
                        case_id=case_id,
                        volume_path=path,
                        metadata={
                            "dataset": "ImageCAS",
                            "source_path": str(path),
                            "kind": "volume",
                        },
                    )
                )
                continue

            if suffix in MESH_EXTENSIONS:
                mesh_case = ImageCASVolumeCase(
                    case_id=case_id,
                    mesh_path=path,
                    metadata={
                        "dataset": "ImageCAS",
                        "source_path": str(path),
                        "kind": "mesh",
                    },
                )
                mesh_cases.append(mesh_case)

        return ImageCASCases(volume_cases=volume_cases, mesh_cases=mesh_cases)
