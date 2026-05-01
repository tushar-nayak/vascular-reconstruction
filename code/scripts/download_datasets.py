"""Download the ImageCAS dataset when access is available."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMAGECAS_KAGGLE_SLUG = "xiaoweixumedicalai/imagecas"


def download_imagecas(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "kagglehub is not installed. Install it in your environment, then rerun this script."
        ) from exc

    dataset_path = kagglehub.dataset_download(IMAGECAS_KAGGLE_SLUG)
    source = Path(dataset_path)
    target = output_dir

    if source.resolve() == target.resolve():
        return

    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download ImageCAS dataset assets for vascular reconstruction.")
    parser.add_argument("--imagecas-dir", type=Path, default=ROOT / "data" / "raw" / "imagecas", help="Target directory for ImageCAS.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    download_imagecas(args.imagecas_dir)


if __name__ == "__main__":
    main()
