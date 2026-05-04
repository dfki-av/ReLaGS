

import os
from scene import Scene
from scene.gaussian_model import GaussianModel
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render


def _build_background_tensor(dataset: ModelParams) -> torch.Tensor:
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    return torch.tensor(bg_color, dtype=torch.float32, device="cuda")


def _build_scene_and_gaussians(dataset: ModelParams, vlm_type: str) -> tuple[Scene, GaussianModel]:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, load_sem=False, vlm_type=vlm_type)
    return scene, gaussians


@torch.no_grad()
def _compute_max_contribution(scene: Scene, gaussians: GaussianModel, pipeline: PipelineParams, background: torch.Tensor) -> None:
    views = scene.getTrainCameras()
    for view in tqdm(views, desc="Calculating Maximum Contribution"):
        render_pkg = render(view, gaussians, pipeline, background)
        gaussians.setMaxContribution(render_pkg["max_contribution"])


def _save_filtered_output(dataset: ModelParams, kept_gaussians: GaussianModel) -> None:
    output_dir = os.path.join(dataset.model_path, "point_cloud", "iteration_0")
    kept_path = os.path.join(output_dir, "point_cloud.ply")
    kept_gaussians.save_ply(kept_path)

@torch.no_grad()
def filter_gaussians(
    dataset: ModelParams,
    pipeline: PipelineParams,
    contribution_threshold: float,
    vlm_type: str,
) -> None:
    scene, gaussians = _build_scene_and_gaussians(dataset, vlm_type)
    background = _build_background_tensor(dataset)

    print("Number of Gaussians before filtering:", gaussians.get_xyz.shape[0])
    _compute_max_contribution(scene, gaussians, pipeline, background)

    mask = gaussians.get_max_contribution > contribution_threshold
    new_gaussians = gaussians.filter_gaussians(mask)

    print("Number of Gaussians after filtering:", new_gaussians.get_xyz.shape[0])    
    _save_filtered_output(dataset, new_gaussians)


def parse_args():
    parser = ArgumentParser(description="Filter Gaussians by max contribution")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--contribution_threshold", type=float, default=0.1)
    parser.add_argument(
        "--vlm_type",
        type=str,
        default="clip",
        choices=["clip", "dinov3", "openseg"],
        help="which vlm type to use",
    )
    args = get_combined_args(parser)
    return model.extract(args), pipeline.extract(args), args


def main() -> None:
    model_args, pipeline_args, args = parse_args()
    filter_gaussians(
        model_args,
        pipeline_args,
        args.contribution_threshold,
        args.vlm_type,
    )


if __name__ == "__main__":
    main()