# 3RScan Data Annotation

This folder contains the offline data preparation scripts for 3RScan scene graph training.

The pipeline has two steps:

1. `scan3r_datapreprocessing.py` renders each scan from sampled RGB camera poses and saves visible object masks per frame.
2. `scan3r_packer.py` reads those masks, extracts per-object visual features, computes object geometry, builds graph edges, and saves packed training pickles.

Run commands from the repository root.

## 1. Object-to-Image Preprocessing

```shell
python scene_graph/data_anno/scan3r_datapreprocessing.py \
  --split train \
  --output_dir /path/to/scan3r_obj2frame \
  --workers 4
```

Output:

```text
/path/to/scan3r_obj2frame/<scan_id>_object2image.pkl
```

Each pickle maps an object id to visible RGB frames:

```text
object_id -> [(image_name, pixel_num, ratio, bbox, rle_mask), ...]
```

Main options:

- `--root`: path to the `3DSSG_subset` metadata folder. Defaults to `<SCAN3R_ROOT_PATH>/3DSSG_subset`.
- `--output_dir`: folder for `*_object2image.pkl` files.
- `--split`: `train`, `validation`, or `test`.
- `--workers`: number of scan-level worker processes.

## 2. Pack Graph Training Data

OpenSeg features:

```shell
python scene_graph/data_anno/scan3r_packer.py \
  --split train \
  --pkl_root /path/to/scan3r_obj2frame \
  --output_dir /path/to/packed_data_openseg \
  --vlm_type openseg \
  --openseg_model_path /path/to/openseg_exported_clip
```

CLIP features:

```shell
python scene_graph/data_anno/scan3r_packer.py \
  --split train \
  --pkl_root /path/to/scan3r_obj2frame \
  --output_dir /path/to/packed_data_clip \
  --vlm_type clip \
  --clip_batch_size 64
```

Output:

```text
<output_dir>/<scan_id>.pkl
```

Packed samples include object geometry features, object visual features, neighbor edges, ground-truth object labels, relationship labels, raw points, and OBB metadata.

Main options:

- `--root`: path to the `3DSSG_subset` metadata folder.
- `--data_root`: path to raw 3RScan scan folders.
- `--pkl_root`: folder created by `scan3r_datapreprocessing.py`.
- `--output_dir`: folder for packed graph pickles.
- `--split`: dataset split to pack.
- `--vlm_type`: `clip` or `openseg`.
- `--openseg_model_path`: required when `--vlm_type openseg`.
- `--clip_batch_size`: crop batch size for CLIP feature extraction.
- `--overwrite`: regenerate existing packed pickles.

