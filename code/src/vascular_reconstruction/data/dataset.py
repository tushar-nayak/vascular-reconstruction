"""Dataset loading for vascular reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
from PIL import Image


class ProjectionDataset:
    """Dataset of multi-view projections (X-rays) and associated geometry."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "dataset.json"
        
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
        
    def __len__(self) -> int:
        return len(self.case_ids)
        
    def get_case(self, index: int) -> dict[str, Any]:
        """Get a specific case by index."""
        case_id = self.case_ids[index]
        views_meta = self.cases_dict[case_id]
        
        views = []
        for meta in views_meta:
            view_path = self.root_dir / meta["image_file"]
            
            view_data = {
                "name": meta["view_name"],
                "image": np.array(Image.open(view_path)),
                "angles": meta["angles_deg"],
                "projection_matrix": meta["projection_matrix"]
            }
            views.append(view_data)
            
        return {
            "case_id": case_id,
            "views": views
        }
