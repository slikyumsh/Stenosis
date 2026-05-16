# Model Scripts

This folder is a curated copy of model-related scripts separated from the main service runtime layout.

## Layout

- `app/` - source mirror for model-facing modules and their import structure
- `tools/` - source mirror for evaluation and comparison utilities
- `yolo/` - quick-access copies for 2D YOLO detection and selected YOLO run configs
- `segmentation_2d/` - quick-access copies for 2D vessel segmentation, local analysis, and drift checks
- `segmentation_3d/` - quick-access copies for 3D SegResNet inference and 3D method comparison
- `evaluation_utils/` - quick-access copies of the most useful evaluation scripts

## Included Topics

- YOLO / 2D detection:
  `app/services/detection_2d.py`
  `tools/eval_2d_detection_bootstrap.py`
  `YOLO_Stenosis_Detection/.../args.yaml`
- 2D segmentation and analysis:
  `app/services/segmentation_2d.py`
  `app/services/analysis_2d.py`
  `app/services/drift_2d.py`
  `tools/measure_2d_subset_metrics.py`
- 3D segmentation and analysis:
  `app/services/segmentation_3d.py`
  `tools/compare_stenosis_local_methods.py`

## Notes

- Dedicated Python training entrypoints for YOLO, 2D segmentation, or 3D segmentation were not found in this repository.
- `YOLO_Stenosis_Detection/` mainly contains run artifacts, weights, and YOLO argument files rather than standalone `train_*.py` scripts.
- Some copied scripts still depend on external assets or neighboring projects, for example:
  `best.onnx`
  SegResNet checkpoints from the configured path
  `../Stenosis3D/scripts` used by `compare_stenosis_local_methods.py`
