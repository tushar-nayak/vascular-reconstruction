# v15 Component Purity Ablation

Baseline selection was done with `code/scripts/score_checkpoints.py` over `single_case_v10` through `single_case_v14`.

Best pre-v15 baseline:

- `single_case_v13` at iteration `1050`
- `largest_component_fraction = 0.704857`
- `component_count = 134`
- `mst_p95 = 3.123290`
- `line_score_mean = 0.704180`

The `v15` ablation resumed from `checkpoints/single_case_v13/checkpoint_1000.pt` and varied only `component_purity_weight`.

Results against the same ranking rule:

- `v15a` (`component_purity_weight = 0.02`) won at iteration `1050`
- `largest_component_fraction = 0.712000`
- `component_count = 124`
- `mst_p95 = 3.124618`
- `line_score_mean = 0.688819`
- structured metrics saved in `code/experiments/v15a_metrics_iter_1050.json`

- `v15b` (`component_purity_weight = 0.05`) increased sampled training connectivity more aggressively, but did not beat `v15a` on the held-out graph diagnostics.
- `v15c` (`component_purity_weight = 0.08`) pushed the sampled component fraction much higher during training, but degraded the final ranking metrics further.

Conclusion:

- a light component-purity penalty improves the main connectivity objective
- stronger weights appear to over-concentrate opacity and hurt overall geometry quality
- the next sweep should stay near `0.02`, not move upward
