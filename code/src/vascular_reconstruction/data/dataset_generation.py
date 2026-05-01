"""Synthetic X-ray and depth-map dataset generation from vessel meshes."""

from __future__ import annotations

import gc
import glob
import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import PIL.Image
import trimesh
from tqdm import tqdm


@dataclass(frozen=True)
class ViewSpec:
    """A single rendering viewpoint."""

    name: str
    lao_angle_deg: float
    cran_angle_deg: float


def default_view_specs() -> tuple[ViewSpec, ...]:
    """Return the default angiography views used for generation."""

    return (
        ViewSpec("AP", 0, 0),
        ViewSpec("Lateral", 90, 0),
        ViewSpec("LAO45", 45, 0),
        ViewSpec("RAO45", -45, 0),
        ViewSpec("Spider", 45, 20),
        ViewSpec("RAO30_Caudal20", -30, -20),
    )


@dataclass
class DatasetGenerationConfig:
    """Configuration for synthetic projection dataset generation."""

    input_dir: str
    output_dir: str
    image_size: int = 2048
    depth_size: int = 1024
    source_pos_mm: tuple[float, float, float] = (-600.0, 0.0, 0.0)
    detector_pos_mm: tuple[float, float, float] = (400.0, 0.0, 0.0)
    detector_size_mm: float = 300.0
    mono_energy_mev: float = 0.06
    photon_count: int = 1000
    vessel_density_g_cm3: float = 2.0
    vessel_compound: str = "H2O"
    mesh_pattern: str = "*_mesh.stl"
    gpu_ids: tuple[int, ...] = (0, 1)
    resume: bool = True
    temp_mesh_prefix: str = "temp_geometry_gpu"
    views: tuple[ViewSpec, ...] = field(default_factory=default_view_specs)

    @property
    def pixel_size_mm(self) -> float:
        return self.detector_size_mm / self.image_size

    @property
    def normalized_gpu_ids(self) -> tuple[int, ...]:
        return self.gpu_ids or (0,)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "image_size": self.image_size,
            "depth_size": self.depth_size,
            "source_pos_mm": list(self.source_pos_mm),
            "detector_pos_mm": list(self.detector_pos_mm),
            "detector_size_mm": self.detector_size_mm,
            "mono_energy_mev": self.mono_energy_mev,
            "photon_count": self.photon_count,
            "vessel_density_g_cm3": self.vessel_density_g_cm3,
            "vessel_compound": self.vessel_compound,
            "mesh_pattern": self.mesh_pattern,
            "gpu_ids": list(self.gpu_ids),
            "resume": self.resume,
            "temp_mesh_prefix": self.temp_mesh_prefix,
            "views": [
                {
                    "name": view.name,
                    "lao_angle_deg": view.lao_angle_deg,
                    "cran_angle_deg": view.cran_angle_deg,
                }
                for view in self.views
            ],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DatasetGenerationConfig":
        raw_views = data.get("views")
        if raw_views is None:
            views = default_view_specs()
        else:
            views = tuple(
                ViewSpec(
                    name=str(view["name"]),
                    lao_angle_deg=float(view["lao_angle_deg"]),
                    cran_angle_deg=float(view["cran_angle_deg"]),
                )
                for view in raw_views
            )
        return cls(
            input_dir=str(data["input_dir"]),
            output_dir=str(data["output_dir"]),
            image_size=int(data.get("image_size", 2048)),
            depth_size=int(data.get("depth_size", 1024)),
            source_pos_mm=tuple(float(v) for v in data.get("source_pos_mm", (-600.0, 0.0, 0.0))),
            detector_pos_mm=tuple(float(v) for v in data.get("detector_pos_mm", (400.0, 0.0, 0.0))),
            detector_size_mm=float(data.get("detector_size_mm", 300.0)),
            mono_energy_mev=float(data.get("mono_energy_mev", 0.06)),
            photon_count=int(data.get("photon_count", 1000)),
            vessel_density_g_cm3=float(data.get("vessel_density_g_cm3", 2.0)),
            vessel_compound=str(data.get("vessel_compound", "H2O")),
            mesh_pattern=str(data.get("mesh_pattern", "*_mesh.stl")),
            gpu_ids=tuple(int(v) for v in data.get("gpu_ids", (0, 1))),
            resume=bool(data.get("resume", True)),
            temp_mesh_prefix=str(data.get("temp_mesh_prefix", "temp_geometry_gpu")),
            views=views,
        )


def load_config(path: str | Path) -> DatasetGenerationConfig:
    """Load a generation config from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return DatasetGenerationConfig.from_mapping(json.load(handle))


def save_config(config: DatasetGenerationConfig, path: str | Path) -> None:
    """Save a generation config to JSON."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(config.to_json_dict(), handle, indent=4)


class XRayDepthGenerator:
    """Generate a multi-view X-ray image and depth map pair for one mesh."""

    def __init__(self, mesh_path: str | Path, output_dir: str | Path, temp_mesh_file: str | Path, config: DatasetGenerationConfig):
        self.original_path = Path(mesh_path)
        self.output_dir = Path(output_dir)
        self.temp_mesh_file = Path(temp_mesh_file)
        self.config = config
        self.base_name = self.original_path.name.replace("_mesh.stl", "")

        self.mesh = trimesh.load(self.original_path, process=False)
        box_center = self.mesh.bounding_box.centroid
        self.mesh.apply_translation(-box_center)

        if np.max(self.mesh.bounding_box.extents) < 50.0:
            self.mesh.apply_scale(10.0)

        self.mesh.export(self.temp_mesh_file)

    def adaptive_contrast_stretch(self, img_array: np.ndarray) -> np.ndarray:
        bg_val = np.percentile(img_array, 95)
        vessel_val = np.min(img_array)

        if (bg_val - vessel_val) < 1e-6:
            return np.full(img_array.shape, 255, dtype=np.uint8)

        img_clipped = np.clip(img_array, vessel_val, bg_val)
        img_norm = (img_clipped - vessel_val) / (bg_val - vessel_val)
        img_gamma = np.power(img_norm, 2.0)
        return (img_gamma * 255).astype(np.uint8)

    def compute_depth_map(self, lao_angle: float, cran_angle: float, filename: str) -> None:
        mesh_copy = self.mesh.copy()

        rot_y = trimesh.transformations.rotation_matrix(np.radians(lao_angle), [0, 1, 0])
        mesh_copy.apply_transform(rot_y)

        rot_z = trimesh.transformations.rotation_matrix(np.radians(cran_angle), [0, 0, 1])
        mesh_copy.apply_transform(rot_z)

        half_w = (self.config.image_size * self.config.pixel_size_mm) / 2.0
        y = np.linspace(-half_w, half_w, self.config.depth_size, dtype=np.float32)
        z = np.linspace(-half_w, half_w, self.config.depth_size, dtype=np.float32)
        yy, zz = np.meshgrid(y, z)

        detector_x = self.config.detector_pos_mm[0]
        pixel_locs = np.stack([np.full_like(yy, detector_x), yy, zz], axis=-1).reshape(-1, 3)
        ray_origins = np.tile(np.asarray(self.config.source_pos_mm, dtype=np.float32), (len(pixel_locs), 1))
        ray_dirs = pixel_locs - ray_origins

        np.divide(ray_dirs, np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis], out=ray_dirs)

        locations, index_ray, _ = mesh_copy.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_dirs,
            multiple_hits=False,
        )

        del pixel_locs, ray_origins, ray_dirs, yy, zz
        gc.collect()

        depth_img = np.zeros(self.config.depth_size * self.config.depth_size, dtype=np.float32)
        if len(locations) > 0:
            source = np.asarray(self.config.source_pos_mm, dtype=np.float32)
            dists = np.linalg.norm(locations - source, axis=1).astype(np.float32)
            min_d, max_d = np.min(dists), np.max(dists)
            if max_d > min_d:
                dists_norm = (dists - min_d) / (max_d - min_d)
                depth_img[index_ray] = dists_norm * 255
            else:
                depth_img[index_ray] = 200

        depth_img = depth_img.reshape(self.config.depth_size, self.config.depth_size).astype(np.uint8)
        img_pil = PIL.Image.fromarray(depth_img).resize((self.config.image_size, self.config.image_size), PIL.Image.LANCZOS)
        img_pil.save(self.output_dir / filename)

    def process_views(self, gvxr: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for view in self.config.views:
            if view.lao_angle_deg != 0:
                gvxr.rotateNode("vessel", view.lao_angle_deg, 0, 1, 0)
            if view.cran_angle_deg != 0:
                gvxr.rotateNode("vessel", view.cran_angle_deg, 0, 0, 1)

            raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
            img_uint8 = self.adaptive_contrast_stretch(raw_xray)

            xray_filename = f"{self.base_name}_{view.name}.png"
            PIL.Image.fromarray(img_uint8).save(self.output_dir / xray_filename)

            depth_filename = f"{self.base_name}_{view.name}_depth.png"
            self.compute_depth_map(view.lao_angle_deg, view.cran_angle_deg, depth_filename)

            f_pix = 1000.0 / self.config.pixel_size_mm
            projection_matrix = [
                [f_pix, 0, self.config.image_size / 2],
                [0, f_pix, self.config.image_size / 2],
                [0, 0, 1],
            ]

            results.append(
                {
                    "image_file": xray_filename,
                    "depth_file": depth_filename,
                    "mesh_source": f"{self.base_name}_mesh",
                    "view_name": view.name,
                    "angles_deg": [view.lao_angle_deg, view.cran_angle_deg],
                    "projection_matrix": projection_matrix,
                }
            )

            if view.cran_angle_deg != 0:
                gvxr.rotateNode("vessel", -view.cran_angle_deg, 0, 0, 1)
            if view.lao_angle_deg != 0:
                gvxr.rotateNode("vessel", -view.lao_angle_deg, 0, 1, 0)

        return results


def _initialize_gvxr(config: DatasetGenerationConfig) -> Any:
    from gvxrPython3 import gvxr

    gvxr.createOpenGLContext()
    gvxr.setSourcePosition(*config.source_pos_mm, "mm")
    gvxr.usePointSource()
    gvxr.setMonoChromatic(config.mono_energy_mev, "MeV", config.photon_count)
    gvxr.setDetectorPosition(*config.detector_pos_mm, "mm")
    gvxr.setDetectorUpVector(0, 0, -1)
    gvxr.setDetectorNumberOfPixels(config.image_size, config.image_size)
    gvxr.setDetectorPixelSize(config.pixel_size_mm, config.pixel_size_mm, "mm")
    return gvxr


def _chunk_mesh_files(mesh_files: list[str], gpu_ids: tuple[int, ...]) -> list[list[str]]:
    return [mesh_files[index::len(gpu_ids)] for index in range(len(gpu_ids))]


def _merge_dataset_parts(output_dir: Path, gpu_ids: tuple[int, ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    dataset_json = output_dir / "dataset.json"

    if dataset_json.exists():
        try:
            with dataset_json.open("r", encoding="utf-8") as handle:
                merged.update(json.load(handle))
        except json.JSONDecodeError:
            pass

    for gpu_id in gpu_ids:
        part_file = output_dir / f"dataset_part_{gpu_id}.json"
        if not part_file.exists():
            continue
        with part_file.open("r", encoding="utf-8") as handle:
            merged.update(json.load(handle))
        part_file.unlink()

    with dataset_json.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=4)

    return merged


def _worker_process(gpu_id: int, mesh_files_chunk: list[str], config: DatasetGenerationConfig) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Initializing worker with {len(mesh_files_chunk)} meshes...")

    gvxr = _initialize_gvxr(config)
    dataset_labels: dict[str, Any] = {}
    temp_mesh_file = Path(config.output_dir) / f"{config.temp_mesh_prefix}{gpu_id}.stl"

    iterable: Iterable[str]
    iterable = tqdm(mesh_files_chunk, position=gpu_id, desc=f"GPU {gpu_id}") if gpu_id == 0 else mesh_files_chunk

    for mesh_path in iterable:
        try:
            generator = XRayDepthGenerator(mesh_path, config.output_dir, temp_mesh_file, config)
            gvxr.loadMeshFile("vessel", str(temp_mesh_file), "mm")
            gvxr.setCompound("vessel", config.vessel_compound)
            gvxr.setDensity("vessel", config.vessel_density_g_cm3, "g/cm3")

            items = generator.process_views(gvxr)
            for item in items:
                key = item.pop("image_file")
                dataset_labels[key] = item

            gvxr.removePolygonMeshesFromSceneGraph()
        except Exception as exc:  # noqa: BLE001
            print(f"[GPU {gpu_id}] Failed on {mesh_path}: {exc}")

    if temp_mesh_file.exists():
        temp_mesh_file.unlink()

    gvxr.terminate()

    worker_json_path = Path(config.output_dir) / f"dataset_part_{gpu_id}.json"
    with worker_json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_labels, handle, indent=4)

    print(f"[GPU {gpu_id}] Finished processing chunk.")


def generate_dataset(config: DatasetGenerationConfig) -> dict[str, Any]:
    """Generate the dataset and return the merged manifest."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(glob.glob(str(Path(config.input_dir) / config.mesh_pattern)))
    if not mesh_files:
        print(f"No meshes found in {config.input_dir}. Please check the folder path.")
        return {}

    total_meshes = len(mesh_files)
    if config.resume:
        pending_meshes = []
        for mesh_path in mesh_files:
            base_name = Path(mesh_path).name.replace("_mesh.stl", "")
            expected_output = output_dir / f"{base_name}_AP.png"
        if not expected_output.exists():
            pending_meshes.append(mesh_path)
        mesh_files = pending_meshes

    if not mesh_files:
        print(f"All {total_meshes} meshes have already been rendered. All done!")
        return _merge_dataset_parts(output_dir, config.normalized_gpu_ids)

    gpu_ids = config.normalized_gpu_ids
    print(
        f"Total pending meshes to render: {len(mesh_files)}. "
        f"Splitting across {len(gpu_ids)} GPU worker(s)..."
    )

    chunks = _chunk_mesh_files(mesh_files, gpu_ids)
    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(target=_worker_process, args=(gpu_id, chunk, config))
        for gpu_id, chunk in zip(gpu_ids, chunks)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
        if worker.exitcode not in (0, None):
            raise RuntimeError(f"Worker process exited with code {worker.exitcode}")

    print("GPU workers finished. Merging JSON datasets...")
    final_dataset = _merge_dataset_parts(output_dir, gpu_ids)
    print("Done. Successfully rendered multi-view data.")
    return final_dataset
