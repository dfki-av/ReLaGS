import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

from configs.scan3r.util_label import *
from configs.scan3r.util_metafile import *
from gaussian_renderer import GaussianModel
from scene_graph.graph_net import SceneGraphEdgeNet
from scene_graph.network_utils import (
    get_edge_geom_features,
    get_oriented_bounding_boxes,
    obb_min_distance_matrix_numpy,
)
from scene_graph.scan3r_dataset import relationship_jina_feature
from scene_graph.network_utils import build_geom_features
from utils.vlm_utils import ClipSimMeasure, DINOV3SimMeasure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = [2, 3]
CHECKPOINT_PATH = "ckpts/graph_transformer_512_80.pth"
MAX_SH_DEGREE = 20
NEIGHBOR_TOPK = 3
EDGE_FEATURE_DIM = 512
VLM_CANON = ["object", "things", "stuff", "texture"]
SH_DEGREE = 3
NUM_LAYERS = 1

REL_NAMES = [
    "none",
    "supported by",
    "left",
    "right",
    "front",
    "behind",
    "close by",
    "inside",
    "bigger than",
    "smaller than",
    "higher than",
    "lower than",
    "same symmetry as",
    "same as",
    "attached to",
    "standing on",
    "lying on",
    "hanging on",
    "connected to",
    "leaning against",
    "part of",
    "belonging to",
    "build in",
    "standing in",
    "cover",
    "lying in",
    "hanging in",
]


def load_model(checkpoint_path: str = CHECKPOINT_PATH) -> SceneGraphEdgeNet:
    model = SceneGraphEdgeNet(L=NUM_LAYERS).to(DEVICE)
    model.eval()
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_path), weights_only=True)["model_state"]
    )
    return model


def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument(
        "--root_pred",
        default="./output/3DSSG_new",
        type=str,
        help="path to save the prediction results",
    )
    parser.add_argument(
        "--vlm_type",
        type=str,
        default="openseg",
        choices=["clip", "dinov3", "openseg"],
        help="which vlm type to use",
    )
    parser.add_argument(
        "--neighbor_thresh",
        type=float,
        default=5.0,
        help="threshold to consider two objects as neighbors",
    )
    parser.add_argument(
        "--edge_batch_size",
        type=int,
        default=1080000 * 2,
        help="batch size for processing edges to limit GPU memory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="path to GNN checkpoint",
    )
    args = parser.parse_args(sys.argv[1:])

    model = load_model(args.checkpoint)

    pred_gs_pth = os.path.join(
        args.root_pred,
        "point_cloud/iteration_{}".format(args.iteration),
        "point_cloud.ply",
    )
    assert os.path.exists(pred_gs_pth), (
        f"Predicted gaussians not found at {pred_gs_pth}"
    )

    gaussians = GaussianModel(SH_DEGREE, MAX_SH_DEGREE)
    gaussians.load_ply(pred_gs_pth)
    points_pred = gaussians._xyz.detach().cpu().numpy()

    if args.vlm_type == "clip":
        nag = torch.load(os.path.join(args.root_pred, "sai_nag.pt"))
        vlm = ClipSimMeasure()
        vlm.load_model()
    elif args.vlm_type == "dinov3":
        nag = torch.load(os.path.join(args.root_pred, "sai_nag_dinov3.pt"))
        vlm = DINOV3SimMeasure()
        vlm.load_model()
    elif args.vlm_type == "openseg":
        nag = torch.load(os.path.join(args.root_pred, "sai_nag_openseg.pt"))
        vlm = ClipSimMeasure(clip_type="ViT-L-14-336")
        vlm.load_model()
    else:
        raise NotImplementedError(f"Unknown vlm type {args.vlm_type}")

    vlm.canon = VLM_CANON

    jina_edges = []
    text_edges = []
    for level in LEVELS:
        cluster_labels = nag["nag"][level].cpu().numpy()
        cluster_clip_feats = nag["nag_feat"][level - 1]
        unique_clusters = np.unique(cluster_labels).astype(np.int32)
        points_pred = points_pred - np.mean(points_pred, axis=0, keepdims=True)
        obb_preds = get_oriented_bounding_boxes(
            points_pred, cluster_labels, unique_clusters, remove_outlier=True
        )

        centers = np.array([info[0] for info in obb_preds.values()])
        Rs = np.array([info[1] for info in obb_preds.values()])
        extents = np.array([info[2] for info in obb_preds.values()])

        node_geom_feat = build_geom_features(centers, Rs, extents).to(DEVICE)
        node_clip_feat = torch.stack(
            [cluster_clip_feats[obj_id] for obj_id in obb_preds.keys()]
        ).to(DEVICE)

        _, adj_mask = obb_min_distance_matrix_numpy(
            centers, Rs, extents, dist_thresh=args.neighbor_thresh, topk=None
        )
        rows, cols = np.nonzero(adj_mask)
        edge_index_np = np.stack([rows, cols])

        centers = torch.tensor(centers, dtype=torch.float32).to(DEVICE)
        Rs = torch.tensor(Rs, dtype=torch.float32).to(DEVICE)
        extents = torch.tensor(extents, dtype=torch.float32).to(DEVICE)

        num_edges = edge_index_np.shape[1]
        all_edge_feat = []
        all_edge_src = []
        all_edge_dst = []

        batch_size = args.edge_batch_size
        num_batches = (num_edges + batch_size - 1) // batch_size
        for start_idx in tqdm(
            range(0, num_edges, batch_size),
            desc=f"Level {level} edges",
            total=num_batches,
        ):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_edge_idx = edge_index_np[:, start_idx:end_idx]
            batch_edge_index = torch.tensor(
                batch_edge_idx, dtype=torch.long, device=DEVICE
            )

            batch_edge_geom_feat = (
                get_edge_geom_features(centers, Rs, extents, batch_edge_index)
                .float()
                .to(DEVICE)
            )
            batch_edge_init_feat = torch.zeros(
                (batch_edge_index.shape[1], EDGE_FEATURE_DIM), dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():
                batch_edge_feat = model(
                    batch_edge_index,
                    batch_edge_init_feat,
                    batch_edge_geom_feat,
                    node_clip_feat,
                    node_geom_feat,
                )
                all_edge_feat.append(batch_edge_feat.cpu())
                all_edge_src.extend(batch_edge_index[0].tolist())
                all_edge_dst.extend(batch_edge_index[1].tolist())

            del (
                batch_edge_index,
                batch_edge_geom_feat,
                batch_edge_init_feat,
                batch_edge_feat,
            )

        edge_feat = torch.cat(all_edge_feat, dim=0).to(DEVICE)
        edge_feat = F.normalize(edge_feat, dim=-1)

        del all_edge_feat
        torch.cuda.empty_cache()

        sims = edge_feat @ relationship_jina_feature.t().to(DEVICE)
        top3 = torch.topk(sims, k=NEIGHBOR_TOPK, dim=-1).indices
        top3_np = top3.cpu().numpy()

        rel_feature = [relationship_jina_feature[i[0]] for i in top3_np]
        rel_cls = [REL_NAMES[i[0]] for i in top3_np]

        jina_dict = dict(zip(zip(all_edge_src, all_edge_dst), rel_feature))
        text_dict = dict(zip(zip(all_edge_src, all_edge_dst), rel_cls))
        jina_edges.append(jina_dict)
        text_edges.append(text_dict)

    torch.save(jina_edges, os.path.join(args.root_pred, "jina_edges_predicted.pt"))
    torch.save(text_edges, os.path.join(args.root_pred, "combined_edges_predicted.pt"))


if __name__ == "__main__":
    main()
