"""Dataset loading for vascular reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ProjectionDataset:
    """Dataset of multi-view projections (X-rays) and associated geometry."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        self.cases = self.manifest.get("cases", [])
        
    def __len__(self) -> int:
        return len(self.cases)
        
    def get_case(self, index: int) -> dict[str, Any]:
        """Get a specific case by index."""
        case = self.cases[index]
        case_id = case["case_id"]
        case_dir = self.root_dir / case_id
        
        # Load views
        views = []
        for view_info in case.get("views", []):
            view_path = case_dir / view_info["image_file"]
            depth_path = case_dir / view_info.get("depth_file", "")
            
            view_data = {
                "name": view_info["name"],
                "image": np.array(Image.open(view_path)),
                "angles": (view_info["lao_angle_deg"], view_info["cran_angle_deg"])
            }
            
            if depth_path.exists():
                view_data["depth"] = np.load(depth_path)
                
            views.append(view_data)
            
        return {
            "case_id": case_id,
            "views": views,
            "mesh_path": self.root_dir / case.get("mesh_file", "")
        }
