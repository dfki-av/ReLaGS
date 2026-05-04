"""
This script performs the *SAM-Guided Graph Edge Reweighting* step in our pipeline.
"""
import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import trace
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.dlpack
import matplotlib.pyplot as plt
import time
import cv2



def f(x):
    return 1 / (1 + torch.relu(x - 5))


@torch.no_grad()
def extract_gaussian_features(model_path, views, gaussians, pipeline, background, feature_level, scale_factor=1):

    for i, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if i == 1: break
        if view.semantic["seg_map"] is None:
            continue
        seg_map = view.semantic["seg_map"][feature_level]
        img_mask = view.semantic['fg_mask'][feature_level]  #.int().cuda()
        seg_min, seg_max = seg_map[img_mask].min().item(), seg_map.max().item()
        seg_num = seg_max - seg_min + 2
        seg_map -= seg_min
        seg_map[~img_mask] = seg_num - 1
        if seg_num > 800:
            print(f"Warning: too many segments {seg_num}")
        enc = torch.normal(mean=0, std=1, size=(seg_num, dim_latent), device="cuda")
        enc = torch.nn.functional.normalize(enc, p=2, dim=-1)
        enc[-1] *= zero_scale

        feature_map = enc[seg_map]
        # ray tracing
        render_pkg = trace(view, gaussians, feature_map, None, pipeline, background)
        gau_depth, gau_sem, num_ray = render_pkg["gaussian_depth"], render_pkg["gaussian_semantics"], render_pkg['num_ray']
        #print(gau_depth.shape, gau_sem.shape, num_ray.shape)
        enc[-1] /= zero_scale
        # gau_sem to label
        gau_sem = torch.nn.functional.normalize(gau_sem, p=2, dim=-1)  # not necessary
        sim = torch.matmul(gau_sem, enc.T)
        sim_max = sim.max(dim=-1)[0]
        gau_sem = sim.argmax(dim=-1)
        sim_filter = sim_max > tau  #### use sim_filter to filter out the unsure points, drop from 0.32 to 0.10
        gau_sem[~sim_filter] = seg_num - 1

        # for each edge, if the two end points are not in the same region, neg_dist increase by 1
        valid_gau = (num_ray > 0) & sim_filter & (gau_sem < seg_num-1)
        depth_weight = f(gau_depth[valid_gau]).unsqueeze(-1)  # TODO: depth_weight
        seen_gau_label = gau_sem[valid_gau].unsqueeze(-1)
        knn_label = gau_sem[knn[valid_gau]]

        neg_dist[valid_gau] += ((knn_label != seen_gau_label) & (knn_label < seg_num-1)).float() * depth_weight * scale_factor
        pos_dist[valid_gau] += ((knn_label == seen_gau_label) & (knn_label < seg_num-1)).float() * depth_weight * scale_factor

def save_colored_knn_similarity(gaussians, knn_clip_similarity, file_name="language_similarity_within_knn_neighbors.ply"):
    """
    Save the similarity as a colored point cloud.
    """
    # Convert knn_clip_similarity to rainbow color
    knn_clip_similarity_mean = knn_clip_similarity.mean(dim=-1).cpu().numpy()
    up_thr = np.percentile(knn_clip_similarity_mean, 95)
    low_thr = np.percentile(knn_clip_similarity_mean, 5)
    knn_clip_similarity_mean = np.clip(knn_clip_similarity_mean, low_thr, up_thr)
    knn_clip_similarity_mean = (knn_clip_similarity_mean - low_thr) / (up_thr - low_thr)
    #knn_clip_similarity_mean = (knn_clip_similarity_mean - knn_clip_similarity_mean.min() 
    knn_clip_similarity_mean = (knn_clip_similarity_mean * 255)
    knn_clip_similarity_color = cv2.applyColorMap(knn_clip_similarity_mean.astype(np.uint8), cv2.COLORMAP_RAINBOW)
    knn_clip_similarity_color = torch.tensor(knn_clip_similarity_color[..., [2,1,0]], dtype=torch.double).cuda()/ 255.0

    # Save new color map
    gaussians._features_dc = knn_clip_similarity_color.view(gaussians._features_dc.shape[0], 1, 3)
    gaussians._features_rest = torch.zeros_like(gaussians._features_rest)

    gaussians.save_ply(os.path.join(args.model_path, file_name))

def process_scene_language_features(dataset : ModelParams, opt : OptimizationParams, iteration : int, pipeline : PipelineParams, feature_level : int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 20)
        scene = Scene(dataset, gaussians, load_iteration=iteration, vlm_type=args.vlm_type)
        gaussians.training_setup(opt)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        scale_factor = 1
        for level in feature_level:
            extract_gaussian_features(args.model_path, scene.getTrainCameras(), gaussians, pipeline, background, level, scale_factor)
            scale_factor *= 0.5
        return gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--in_neigh", default="neighbor.pt")
    parser.add_argument("--out_neigh", default="neighbor_new.pt")
    parser.add_argument("--config", default="configs/def.yaml")
    parser.add_argument("--level", default=1, type=int, nargs='+')
    parser.add_argument('--vlm_type', type=str, default='clip', choices=['clip', 'dinov3', 'openseg'],
                        help='which vlm type to use')
    args = get_combined_args(parser)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config).graph_weight

    model_path = model.extract(args).model_path
    neighbors = torch.load(os.path.join(model_path, args.in_neigh))
    knn = neighbors['neighbors'].cuda()
    ori_distance = neighbors['distances'].cuda()
    neg_dist = torch.zeros_like(knn, dtype=torch.float).cuda()
    pos_dist = torch.zeros_like(knn, dtype=torch.float).cuda()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dim_latent = 20
    tau, zero_scale = cfg.tau, cfg.zero_scale
    neg_w, pos_w = cfg.neg_w, cfg.pos_w
    neg_b, pos_b = cfg.neg_b, cfg.pos_b

    gaussians = process_scene_language_features(model.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.level)
    # increase the distance of the knn, if the two end points are not in the same region
    neg_dist = torch.clamp(neg_dist, 0, neg_b)
    pos_dist = torch.clamp(pos_dist, 0, pos_b)
    new_distance = ori_distance + neg_dist * neg_w - pos_dist * pos_w
    new_distance = torch.clamp(new_distance, min=0)

    torch.save({'neighbors': knn, 'distances': new_distance}, os.path.join(model_path, args.out_neigh))
    #save_colored_knn_similarity(gaussians, new_distance, file_name="reweight_knn_neighbors.ply")
    #save_colored_knn_similarity(gaussians, ori_distance, file_name="original_knn_neighbors.ply")

    #import matplotlib.pyplot as plt
    '''# Visualize the new knn distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(new_distance.cpu().numpy().flatten(), bins=100, density=True)
    plt.title('Distribution of New KNN Distance')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid()
    plt.savefig(os.path.join(args.model_path, "new_knn_distance_distribution.png"))
    plt.close()

    # Visualize the original knn distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ori_distance.cpu().numpy().flatten(), bins=100, density=True)
    plt.title('Distribution of Original KNN Distance')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid()
    plt.savefig(os.path.join(args.model_path, "original_knn_distance_distribution.png"))
    plt.close()
    
    '''
