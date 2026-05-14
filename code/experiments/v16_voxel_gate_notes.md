# v16 Voxel Gate / Reduced Growth

This round changed evaluation and training in five ways:

- added voxel and mesh-aware reconstruction evaluation
- added hard gate thresholds evaluated during training
- ranked checkpoints with voxel connectivity and mesh distance, not only kNN graph proxies
- added a voxel-topology loss and disabled the weaker component-purity term in `v16`
- reduced active Gaussian growth to a maximum of `8000`

Strict baseline under the new ranking rule:

- `single_case_v13` at iteration `1200`
- `voxel_largest_component_fraction = 0.406`
- `voxel_component_count = 1070`
- `mesh_vertex_chamfer_p95 = 200.478`
- `largest_component_fraction = 0.389`

`v16` configuration:

- resumed from `checkpoints/single_case_v13/checkpoint_1000.pt`
- enabled `voxel_topology_weight = 0.18`
- reduced active schedule to `4500 -> 6000 -> 7000 -> 8000`
- gate thresholds required:
- `graph_largest_component_fraction >= 0.45`
- `graph_component_count <= 300`
- `voxel_largest_component_fraction >= 0.45`
- `voxel_component_count <= 900`
- `occupancy_fill_ratio <= 0.03`
- `mesh_vertex_chamfer_p95 <= 180.0`

Observed result:

- every `v16` eval failed the hard gate
- `v16` stayed around `voxel_largest_component_fraction ~ 0.33`
- `v16` stayed around `voxel_component_count ~ 1440 - 1481`
- `v16` stayed around `mesh_vertex_chamfer_p95 ~ 198.4 - 198.6`
- strict rescoring kept `single_case_v13 @ 1200` as the best checkpoint

Conclusion:

- the new gate is useful because it rejects reconstructions that still look poor in volume and mesh space
- reduced growth pressure plus the current voxel-topology penalty is not yet enough to improve the reconstruction
- the next step should target the geometry representation or supervision itself, not just regularizer weight tuning
