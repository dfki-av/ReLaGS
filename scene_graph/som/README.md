# SoM Scene Graph Pipeline

This folder contains the Set-of-Mark (SoM) scene graph workflow:

1. `scene_graph_som_gpt_annotation.py` renders numbered object masks for each training view and asks ChatGPT to produce per-view 2D object and relationship annotations.
2. `build_3dsg_multi_level.py` lifts those 2D annotations onto 3D superpoints, merges edges across views, and optionally converts relationship text to Jina embeddings.

Run commands from the repository root.

## 1. Generate SoM ChatGPT Annotations

```shell
python scene_graph/som/scene_graph_som_gpt_annotation.py \
  --model_path ./output/lerf_ovs/teatime \
  --source_path ./datasets/lerf_ovs/teatime \
  --iteration 0 \
  --data_dir ./datasets/lerf_ovs/teatime \
  --vlm_type clip \
  --out_dir chatgpt
```

Before running, set your OpenAI API key:

```shell
set OPEN_API_KEY=your_api_key
```

On Linux/macOS:

```shell
export OPEN_API_KEY=your_api_key
```

The script creates SoM images and GPT annotation files in:

```text
<data_dir>/<out_dir>/
```

The next script currently reads annotations from `<model_path>/chatgpt/`, not from `data_dir`. If `data_dir` and `model_path` are different, copy or sync the generated JSON files before lifting:

```text
<model_path>/chatgpt/
```

Important options:

- `--model_path`: trained Gaussian scene output folder.
- `--source_path`: source dataset path used by the Gaussian scene loader.
- `--data_dir`: dataset folder containing `images/`.
- `--iteration`: checkpoint iteration to load.
- `--vlm_type`: one of `clip`, `dinov3`, or `openseg`; this selects the matching `sai_nag*.pt` file from `model_path`.
- `--rel_types`: `semantic` or `affordance`.
- `--redo`: comma-separated image names to regenerate.
- `--edit`: reuse an edited GPT text output during redo.
- `--debug`: process only a small number of frames.

Expected annotation JSON fields:

```json
{
  "objects": {},
  "relationships": [],
  "object_map": {}
}
```

`object_map` maps each GPT object id to the rendered SoM/SAM id used by the 3D lifting step. If the GPT object ids are already the numbered SoM labels, this can be an identity map.

## 2. Lift 2D Annotations to a Multi-Level 3D Scene Graph

```shell
python scene_graph/som/build_3dsg_multi_level.py \
  --model_path ./output/lerf_ovs/teatime \
  --source_path ./datasets/lerf_ovs/teatime \
  --iteration 0 \
  --vlm_type clip \
  --output_folder ./scene_graph_temp_imgs
```

This script reads 2D annotations from:

```text
<model_path>/chatgpt/
```

It writes per-view lifted edges to:

```text
<output_folder>/<scene_name>/<frame_name>/edges.pt
```

It writes merged multi-level text edges to:

```text
<model_path>/combined_edges_ml.pt
```

If Jina conversion is enabled, it also writes:

```text
<model_path>/jina_edges_ml.pt
```

Important options:

- `--skip_lift`: skip 2D-to-3D lifting and only run later stages.
- `--skip_merge`: skip merging per-view edges.
- `--skip_jina`: skip relationship-text embedding with Jina.
- `--fill_knn`: enable local KNN edge filling when supported by the current code path.
- `--output_folder`: folder for temporary per-view scene graph edges.
- `--anno_folder`: currently parsed but not used; annotations are read from `<model_path>/chatgpt/`.
- `--vlm_type`: must match the NAG file available in `model_path`.

## Typical Two-Step Run

```shell
python scene_graph/som/scene_graph_som_gpt_annotation.py \
  --model_path ./output/lerf_ovs/teatime \
  --source_path ./datasets/lerf_ovs/teatime \
  --iteration 0 \
  --data_dir ./datasets/lerf_ovs/teatime \
  --vlm_type clip \
  --out_dir chatgpt

# Make annotations visible to the lifting script if needed:
# copy ./datasets/lerf_ovs/teatime/chatgpt/*.json ./output/lerf_ovs/teatime/chatgpt/

python scene_graph/som/build_3dsg_multi_level.py \
  --model_path ./output/lerf_ovs/teatime \
  --source_path ./datasets/lerf_ovs/teatime \
  --iteration 0 \
  --vlm_type clip \
  --output_folder ./scene_graph_temp_imgs
```

Use `--skip_jina` if you only need text relationships or do not have the Jina dependencies installed.
