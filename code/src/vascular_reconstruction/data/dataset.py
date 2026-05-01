"""Dataset loading for vascular reconstruction with Distance Transform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


class ProjectionDataset:
    """Dataset of multi-view projections (X-rays) and associated geometry."""

    def __init__(self, root_dir: str | Path, compute_dt: bool = True):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "dataset.json"
        self.compute_dt = compute_dt
        
        if not self.manifest_path.exists():
            if (self.root_dir / "manifest.json").exists():
                self.manifest_path = self.root_dir / "manifest.json"
            else:
                raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        # Group by mesh_source
        self.cases_dict = defaultdict(list)
        for img_file, meta in self.manifest.items():
            meta["image_file"] = img_file
            self.cases_dict[meta["mesh_source"]].append(meta)
            
        self.case_ids = sorted(list(self.cases_dict.keys()))
        self.dt_cache = {} # Cache for distance transforms
        
    def __len__(self) -> int:
        return len(self.case_ids)
        
    def get_case(self, index: int) -> dict[str, Any]:
        """Get a specific case by index."""
        case_id = self.case_ids[index]
        views_meta = self.cases_dict[case_id]
        
        views = []
        for meta in views_meta:
            view_path = self.root_dir / meta["image_file"]
            
            image_np = np.array(Image.open(view_path))
            
            view_data = {
                "name": meta["view_name"],
                "image": image_np,
                "angles": meta["angles_deg"],
                "projection_matrix": meta["projection_matrix"]
            }
            
            if self.compute_dt:
                # Precompute or retrieve Distance Transform of the vessel mask
                # Vessels are dark (< 128 roughly), background is light (~255)
                # We want DT of the silhouette.
                vessel_mask = (image_np < 200).astype(np.float32)
                
                dt_key = meta["image_file"]
                if dt_key not in self.dt_cache:
                    # DT is distance to the nearest NON-vessel pixel?
                    # No, we want distance to the nearest VESSEL pixel for points outside.
                    # distance_transform_edt computes distance to the nearest ZERO pixel.
                    # So for points outside, we want distance to vessel (where mask=1).
                    # edt(1 - vessel_mask) gives distance to the vessel.
                    dt = distance_transform_edt(1.0 - vessel_mask)
                    self.dt_cache[dt_key] = dt.astype(np.float32)
                
                view_data["distance_transform"] = self.dt_cache[dt_key]
                
            views.append(view_data)
            
        return {
            "case_id": case_id,
            "views": views
        }
