"""
This script performs the superpoint merging and semantic feature reprojection of our pipeline.
It is derived and optimized from the SAI3D paper (https://arxiv.org/abs/2312.11557).
"""

import argparse
from os.path import join
import time
from collections import deque

import numpy as np
import torch
import scipy.spatial
from tqdm import tqdm

import utils.sai3d_utils as utils
from utils.linetimer import CodeTimer

from scene import Scene, GaussianModel
from gaussian_renderer import render_point, trace
from arguments import *
import torch.nn.functional as F


N_WORKERS = 20


class SAI3D:
    def __init__(self, args):
        self.max_neighbor_distance = args.max_neighbor_distance
        self.args = args
        self.view_freq = args.view_freq
        self.dis_decay = args.dis_decay
        self._pipeline = None
        self._background = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = PipelineParams(ArgumentParser())
        return self._pipeline

    @property
    def background(self):
        if self._background is None:
            self._background = torch.tensor(
                [0, 0, 0], dtype=torch.float32, device="cuda"
            )
        return self._background

    def init_data(self, args):
        self.gaussians = GaussianModel(3, 0)
        start = time.time()
        print("load iteration:", args.iteration)
        self.scene = Scene(
            args, self.gaussians, load_iteration=args.iteration, vlm_type=args.vlm_type
        )
        print("Load Scene Imgs:", time.time() - start)
        self.N = self.scene.gaussians._xyz.shape[0]
        self.seg_ids = torch.load(join(args.model_path, "nag-l1.pt"))
        self.points = self.scene.gaussians._xyz.cpu().detach().numpy()
        self.M = len(self.scene.getTrainCameras())
        start = time.time()
        points_kdtree = scipy.spatial.KDTree(self.points)
        points_neighbors = points_kdtree.query(self.points, 8, workers=N_WORKERS)[
            1
        ]  # (n,k)
        self.points_neighbors = torch.tensor(points_neighbors, dtype=torch.long).cuda()
        print("Time kdtree:", time.time() - start)
        print(
            "Points num:",
            self.N,
            ", views num:",
            self.M,
            ", sp num:",
            self.seg_ids.max().item() + 1,
        )

    def _get_superpoint_mask_ratio(self, point_to_super, point_to_mask, num_masks):
        """Compute assignment ratio between superpoints and mask regions."""
        num_superpoints = point_to_super.max().item() + 1

        # Efficient bincount with flattened indices
        indices = point_to_super * num_masks + point_to_mask
        super_mask_count = torch.bincount(
            indices, minlength=num_superpoints * num_masks
        ).view(num_superpoints, num_masks)

        # Remove background (mask=0)
        if point_to_mask.min() == 0:
            super_mask_count = super_mask_count[:, 1:]

        super_mask_sum = super_mask_count.sum(dim=1, keepdim=True)
        return super_mask_count.float() / (super_mask_sum + 1e-8)

    def _filter_feature_outliers(self, features, device: str = "cuda") -> torch.Tensor:
        """Filter outlier features based on cosine similarity statistics. (ROFA)

        This is our main contribution: using statistical outlier detection to remove
        inconsistent feature observations before aggregation, improving robustness.

        Args:
            features: Tensor of shape (N, D) containing N feature vectors of dimension D
            device: Device to create tensors on

        Returns:
            Normalized aggregated feature vector of shape (D,)
        """
        feat_dim = features.shape[1]

        if features.shape[0] == 0:
            return torch.zeros(feat_dim, device=device)

        if features.shape[0] == 1:
            return F.normalize(features[0], p=2, dim=-1)

        cos_sim = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        )

        mean_sim = (cos_sim.sum(dim=1) - 1) / (cos_sim.shape[0] - 1)

        mu, sigma = mean_sim.mean(), mean_sim.std()
        keep_mask = mean_sim > (mu - self.args.tau * sigma)

        if keep_mask.sum() == 0:
            keep_mask[mean_sim.argmax()] = True

        filtered_feats = features[keep_mask]
        return F.normalize(filtered_feats.mean(dim=0), p=2, dim=-1)

    @torch.no_grad()
    def proj_gaussian_features(
        self, gaussians, views, feature_level, gau2sp, pt_sp_label, device="cuda"
    ):
        """Project and aggregate semantic features to superpoints with outlier filtering."""

        WEIGHT_THRESHOLD = 0.0001
        RATIO_THRESHOLD = 0.3

        sp_uni, sp_gnum = torch.unique(gau2sp, return_counts=True, sorted=True)
        feat_dim = views[0].semantic["sem"].to(device).shape[1]
        sp_feature_list = [[] for _ in range(sp_uni.shape[0])]

        pipeline = self.pipeline
        background = self.background

        # Render and accumulate features from each view
        for view_idx, view in enumerate(tqdm(views, desc="Projection progress")):
            render_pkg = render_point(view, gaussians, pipeline, background)
            weight = render_pkg["weight"]
            gau_mask = weight > WEIGHT_THRESHOLD

            gt_feature = view.semantic["sem"].to(device)
            seg_map = view.semantic["seg_map"][feature_level]
            img_mask = view.semantic["fg_mask"][feature_level]

            seg_min = seg_map[img_mask].min().item()
            seg_max = seg_map[img_mask].max().item()
            view_level_feature = gt_feature[seg_min : seg_max + 1]
            seg_num = seg_max - seg_min + 2

            # Compute superpoint-to-mask assignment ratios
            sp_mask_mat = self._get_superpoint_mask_ratio(
                gau2sp, pt_sp_label[view_idx], seg_num
            )
            sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0
            sp_feat = sp_mask_mat @ view_level_feature

            # Accumulate weighted features for visible superpoints
            sp_index = gau2sp[gau_mask]
            unique_sp, sp_gau_count = torch.unique(sp_index, return_counts=True)

            for sp_idx, sp in enumerate(unique_sp):
                visibility_portion = sp_gau_count[sp_idx] / sp_gnum[sp].item()
                feat = F.normalize(sp_feat[sp], p=2, dim=-1) * visibility_portion
                if feat.any():
                    sp_feature_list[sp].append(feat)

        # Filter outliers and compute final features using our statistical method
        sp_features = []
        for sp in range(len(sp_feature_list)):
            if sp_feature_list[sp]:
                features = torch.stack(sp_feature_list[sp], dim=0)
                
                # outlier-aware feature aggregation
                sp_features.append(self._filter_feature_outliers(features, device))
            else:
                sp_features.append(torch.zeros(feat_dim, device=device))

        return F.normalize(torch.stack(sp_features, dim=0), p=2, dim=-1)

    def extract_gaussian_features(self, gaussians, views, feature_level):
        pipeline = self.pipeline
        background = self.background
        labels = []
        for i, view in enumerate(tqdm(views, desc="Rendering progress")):
            seg_map = view.semantic["seg_map"][feature_level].clone()
            img_mask = view.semantic["fg_mask"][feature_level]
            seg_min, seg_max = (
                seg_map[img_mask].min().item(),
                seg_map[img_mask].max().item(),
            )
            seg_num = seg_max - seg_min + 2
            seg_map -= seg_min - 1
            seg_map[~img_mask] = 0
            enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
            enc = torch.nn.functional.normalize(enc, p=2, dim=-1)

            feature_map = enc[seg_map]
            # ray tracing
            render_pkg = trace(view, gaussians, feature_map, None, pipeline, background)
            gau_depth, gau_sem, num_ray = (
                render_pkg["gaussian_depth"],
                render_pkg["gaussian_semantics"],
                render_pkg["num_ray"],
            )
            # gau_sem to label
            gau_sem = torch.nn.functional.normalize(
                gau_sem, p=2, dim=-1
            )  # not necessary
            sim = torch.matmul(gau_sem, enc.T)
            gau_sem = sim.argmax(dim=-1)
            sim_filter = sim.max(dim=-1)[0] > 0.85
            gau_sem[~sim_filter] = 0
            labels.append(gau_sem)
        return torch.stack(labels).transpose(0, 1)  # (N, M)

    def get_seg_data_torch(self):
        seg_ids = utils.num_to_natural_torch(self.seg_ids)
        unique_seg_ids, counts = torch.unique(seg_ids, return_counts=True)
        seg_num = unique_seg_ids.shape[0]

        sorted_indices = torch.argsort(seg_ids)
        split_points = counts.cumsum(0)[:-1]
        seg_members_list = torch.tensor_split(sorted_indices, split_points.tolist())
        seg_members = {
            seg_id.item(): members
            for seg_id, members in zip(unique_seg_ids, seg_members_list)
        }

        # 1. find neighboring points of each point

        # 2. find directly neighboring superpoints of each superpoint with the help of point neighbors
        seg_direct_neighbors = torch.zeros((seg_num, seg_num), dtype=torch.bool)
        for id, members in seg_members.items():
            neighbors = self.points_neighbors[members]
            neighbor_seg_ids = seg_ids[neighbors]
            seg_direct_neighbors[id][neighbor_seg_ids] = 1
        seg_direct_neighbors[torch.eye(seg_num, dtype=torch.bool)] = 0  # exclude self
        # make neighboring matrix symmetric
        seg_direct_neighbors[seg_direct_neighbors.clone().T] = 1

        # 3. find indirectly neighboring superpoints of each superpoint
        # zeroth dimension is "distance" of two superpoints
        seg_indirect_neighbors = torch.zeros(
            (self.max_neighbor_distance, seg_num, seg_num),
            dtype=torch.bool,
            device=seg_ids.device,
        )
        seg_indirect_neighbors[0] = seg_direct_neighbors
        for i in range(
            1, self.max_neighbor_distance
        ):  # to get neighbors with distance leq i+1
            for seg_id in range(seg_num):
                last_layer_neighbors = seg_indirect_neighbors[i - 1, seg_id]
                this_layer_neighbors = (
                    seg_indirect_neighbors[i - 1, last_layer_neighbors].sum(0) > 0
                )
                seg_indirect_neighbors[i, seg_id] = this_layer_neighbors
            # exclude self
            seg_indirect_neighbors[i, torch.eye(seg_num, dtype=torch.bool)] = 0
            # include closer neighbors
            seg_indirect_neighbors[i, seg_indirect_neighbors[i - 1].clone()] = 1

        self.seg_member_count = counts.cuda()
        return seg_ids, seg_num, seg_members, seg_indirect_neighbors

    @torch.no_grad()
    def assign_label(self, thres_connect, max_neighbor_distance=2):
        """assign instance labels for all points in the scene

        :param points: (N, 3), points in world coordinate
        :param thres_connect: the threshold for judging whether two superpoints are connected
        :param max_neighbor_distance: the max logical distance of indirect neighbors to take into account
        :param similar_metric: the metric to calculate the similarity between two primitives
        :return pt_prim_label: (N,), resulting instance labels of all points
        """

        points_seen = None  # no need
        pt_prim_label = None
        nag = [self.seg_ids]
        nag_features = []

        # progressive region growing,
        # the resulting oversegmentations of last iteration can be the primitive of next iteration.
        for i in range(len(thres_connect)):
            pre_time = time.time()

            pt_sp_label = self.extract_gaussian_features(
                self.gaussians, self.scene.getTrainCameras(), i + 1
            )
            if self.args.seg_enhance and i == 0:
                upper_label = self.extract_gaussian_features(
                    self.gaussians, self.scene.getTrainCameras()[::3], i + 2
                )
                pt_sp_label = torch.cat([pt_sp_label, upper_label], dim=1)

            print("get_points_label_and_seen:", time.time() - pre_time)
            (
                self.seg_ids,
                self.seg_num,
                self.seg_members,
                self.seg_indirect_neighbors,
            ) = self.get_seg_data_torch()
            self.seg_direct_neighbors = self.seg_indirect_neighbors[0]
            seg_adj = self.get_seg_adjacency(
                points_label=pt_sp_label,
                points_seen=points_seen,
            )
            seg_labels = self.assign_seg_label_torch(
                seg_adj, thres_connect[i], max_neighbor_distance=max_neighbor_distance
            )

            # only conduct postprocessing in the last iteration
            if i == len(thres_connect) - 1 and self.args.thres_merge > 0:
                seg_labels = self.merge_small_segs_torch(
                    seg_labels, self.args.thres_merge, seg_adj
                )
                seg_labels = utils.num_to_natural_torch(seg_labels) + 1

            # assign primitive labels to member points
            pt_prim_label = torch.zeros(
                self.N, dtype=torch.int, device=seg_labels.device
            )
            for j in range(self.seg_num):
                label = seg_labels[j]
                if j in self.seg_members:
                    pt_prim_label[self.seg_members[j]] = label
                else:
                    print(f"Warning: Key {j} not found in self.seg_members")
            nag.append(pt_prim_label - 1)
            self.seg_ids = pt_prim_label

            # project the semantic features of superpoints to the points
            sp_feature = self.proj_gaussian_features(
                self.gaussians,
                self.scene.getTrainCameras(),
                i + 1,
                pt_prim_label - 1,
                pt_sp_label.T,
            )
            sp_feature = F.normalize(sp_feature, p=2, dim=-1)
            nag_features.append(sp_feature)
            torch.cuda.empty_cache()

        # count invalid points
        invalid_count = (pt_prim_label == -1).sum()
        print("number of invalid points:", invalid_count.item())
        print(
            f"nag shape: {[x.max().item() + 1 for x in nag]}"
        )  # all labels start from 0, and are continuous
        return pt_prim_label, nag, nag_features

    def assign_seg_label_torch(
        self,
        adj: torch.Tensor,
        thres_connect,
        max_neighbor_distance,
        dense_neighbor=False,
    ):
        """Optimized region growing algorithm for superpoints using BFS."""
        pre_time = time.time()

        assign_id = 1
        seg_labels = torch.zeros(self.seg_num, dtype=torch.int, device=adj.device)

        for i in range(self.seg_num):
            if seg_labels[i] == 0:
                queue = deque([i])
                seg_labels[i] = assign_id
                group_points_count = self.seg_member_count[i].clone()

                while queue:
                    v = queue.popleft()
                    js = (
                        self.seg_direct_neighbors[v].nonzero(as_tuple=True)[0]
                        if not dense_neighbor
                        else self.seg_direct_neighbors[v]
                    )

                    # Only select unlabeled neighbors
                    js = js[seg_labels[js] == 0]

                    # Check only pre-filtered candidates
                    for j in js:
                        connect = self.judge_connect_torch_opt(
                            adj,
                            j,
                            thres_connect,
                            seg_labels,
                            assign_id,
                            max_neighbor_distance,
                            decay=self.dis_decay,
                        )

                        if connect:
                            seg_labels[j] = assign_id
                            group_points_count += self.seg_member_count[j]
                            queue.append(j)

                assign_id += 1

        print("time for region_growing:", time.time() - pre_time)
        return seg_labels  # (s, )

    """
    The three functions below are used to calculate the adjacency when the primitive of growing is superpoints.
    """

    def get_seg_adjacency(self, points_label, points_seen):
        """
        :params similar_meric: the metric to calculate the similarity between two primitives
        :params points_label: (N, M), labels of all points in all views
        :params points_seen: (N, M), seen flag of all points in all views

        :return: adjacency_mat, (s,s): adjacency between each pair of neighboring segs
        """
        pre_time = time.time()
        similar_mat, confidence_mat = self.get_neighbor_seg_similar_confidence_matrix(
            points_label, points_seen, self.max_neighbor_distance, self.args.thres_trunc
        )
        adjacency_mat = self.get_seg_adjacency_from_similar_confidence_torch(
            similar_mat, confidence_mat
        )
        print("get_seg_adjacency_from_score:", time.time() - pre_time)
        return adjacency_mat

    def get_neighbor_seg_similar_confidence_matrix(
        self,
        points_label,
        points_seen,
        max_neighbor_distance,
        thres_trunc=0.0,
    ):
        """

        :param points_label: (N, M), labels of all points in all views
        :param points_seen: (N, M), seen flag of all points in all views
        :param max_neighbor_distance: the max logical distance of indirect neighbors to take into account
        :param thres_trunc: the threshold for discarding the similarity between two primitives if their confidence is too low
        :return similar_sum (s,s): weight sum of similar score in every view
        :return confidence (s,s): sum of confidence of how much we can trust the similar score in every view
        """
        seg_neighbors = self.seg_indirect_neighbors[
            max_neighbor_distance - 1
        ]  # binary matrix, (s,s)
        seg_members = self.seg_members  # dict {seg_id: point_array}
        seg_ids = self.seg_ids

        seg_seen0 = torch.zeros(
            [self.seg_num, points_label.shape[1]], dtype=torch.float32
        )
        for seg_id, members in seg_members.items():
            seg_seen0[seg_id] = (
                (points_label[members] > 0).sum(axis=0)
            ) / members.shape[0]  # (mem,m) -sum-> (m,)

        similar_sum, confidence_sum = utils.torch_get_similar_confidence_matrix(
            seg_neighbors, seg_ids, seg_seen0, points_label, thres_trunc
        )

        return similar_sum, confidence_sum

    def get_seg_adjacency_from_similar_confidence_torch(
        self, similar_mat, confidence_mat, chunk_size=15000
    ):
        assert similar_mat.nonzero().size(0) > 0
        adj = torch.zeros([self.seg_num, self.seg_num], device=similar_mat.device)
        r, c = confidence_mat.nonzero(as_tuple=True)
        total = r.numel()
        for i in range(0, total, chunk_size):
            idx_slice = slice(i, min(i + chunk_size, total))
            ri = r[idx_slice]
            ci = c[idx_slice]

            val = similar_mat[ri, ci] / confidence_mat[ri, ci]
            adj[ri, ci] = val

        r2, c2 = adj.nonzero(as_tuple=True)
        adj[r2, c2] = adj[c2, r2] = torch.max(adj[r2, c2], adj[c2, r2])
        return adj

    def judge_connect_torch_opt(
        self,
        adj,
        p2_id,
        thres_connect,
        seg_labels,
        region_label,
        max_neighbor_distance,
        decay=0.5,
    ):
        weight = decay ** torch.arange(max_neighbor_distance, device=adj.device)

        neighbors_at_dist = self.seg_indirect_neighbors[
            :, p2_id
        ]  # (max_neighbor_distance, seg_num)
        label_mask = (seg_labels == region_label).unsqueeze(0)  # (1, seg_num)
        valid_neighbors = (
            neighbors_at_dist & label_mask
        )  # (max_neighbor_distance, seg_num)

        # Calculate score
        neighbor_ids = valid_neighbors.nonzero(as_tuple=True)[1]
        adj_sum = (
            weight[:, None]
            * adj[p2_id, neighbor_ids]
            * self.seg_member_count[neighbor_ids]
        ).sum()
        weight_sum = (weight[:, None] * self.seg_member_count[neighbor_ids]).sum()

        score = adj_sum / weight_sum if weight_sum != 0 else 0
        return score >= thres_connect

    def merge_small_segs_torch(
        self, seg_labels: torch.Tensor, merge_thres, adj: torch.Tensor
    ):
        """postprocess segmentation results by merging small regions into neighbor regions with high affinity

        :param seg_labels: (s,), resulting labels of all superpoints
        :param merge_thres: threshold for filtering small regions
        :param adj: (s, s), affinity matrix between each pair of neighboring superpoints

        """
        seg_member_count = self.seg_member_count
        unique_labels, seg_count = torch.unique(seg_labels, return_counts=True)
        region_num = unique_labels.shape[0]

        merged_labels = seg_labels.clone()
        merged_mask = torch.ones_like(seg_labels)
        merge_count = 0

        for i in range(region_num):
            if seg_count[i] > 2:
                continue
            label = unique_labels[i]
            seg_ids = (seg_labels == label).nonzero(as_tuple=True)[0]
            if seg_member_count[seg_ids].sum() < merge_thres:
                merged_mask[seg_ids] = 0

        finished = False
        while not finished:
            flag = False
            for i in range(region_num):
                label = unique_labels[i]
                seg_ids = (seg_labels == label).nonzero(as_tuple=True)[0]
                if merged_mask[seg_ids[0]] > 0:
                    continue
                seg_sims = adj[seg_ids].sum(dim=0)
                adj_sort = torch.argsort(seg_sims, descending=True)

                for j in range(adj_sort.shape[0]):
                    target_seg_id = adj_sort[j]
                    if merged_mask[target_seg_id] == 0:
                        continue
                    if seg_sims[target_seg_id] == 0:
                        break
                    target_label = merged_labels[target_seg_id]
                    merged_labels[seg_ids] = target_label.clone()
                    merged_mask[seg_ids] = 1
                    merge_count += 1
                    flag = True
                    break
            if not flag:
                finished = True

        merged_labels[merged_mask == 0] = 0
        print(
            "original region number:",
            seg_count.shape[0],
            "merging count:",
            merge_count,
            "remove count:",
            (merged_mask == 0).sum().item(),
        )
        return merged_labels


def everything_seg(args):
    time_collection = {}

    agent = SAI3D(args)

    agent.init_data(args)

    with CodeTimer("Assign instance labels", dict_collect=time_collection):
        labels_fine_global, nag, nag_feat = agent.assign_label(
            thres_connect=args.thres_connect,
            max_neighbor_distance=args.max_neighbor_distance,
        )

    with CodeTimer("Save results", dict_collect=time_collection):
        suffix_map = {
            "clip": f"_{args.tau}",
            "dinov3": "_dinov3",
            "openseg": "_openseg",
        }
        suffix = suffix_map.get(args.vlm_type, "")
        torch.save(
            {"nag": nag, "nag_feat": nag_feat},
            join(args.model_path, f"sai_nag{suffix}.pt"),
        )
        utils.save_label_ply(
            agent.points,
            labels_fine_global,
            join(args.model_path, args.model_path.split("/")[-2] + "saifine.ply"),
        )
    print("Final labels num:", torch.unique(labels_fine_global).shape[0])

    for k, v in time_collection.items():
        print(f"Time {k}: {v:.1f}")
    print(f"Total time: {sum(time_collection.values()):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument(
        "--base_dir", type=str, default="data/ScanNet", help="path to scannet dataset"
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="just a case for tweak parameter, will save file in particular names",
    )
    parser.add_argument(
        "--view_freq",
        type=int,
        default=5,
        help="how many views to select one view from",
    )
    parser.add_argument(
        "--thres_connect",
        type=str,
        default="0.5",
        help='dynamic threshold for progresive region growing, in the format of "start_thres,end_thres,stage_num',
    )
    parser.add_argument(
        "--dis_decay",
        type=float,
        default=0.5,
        help="weight decay for calculating seg-region affinity",
    )
    parser.add_argument(
        "--thres_dis",
        type=float,
        default=0.15,
        help="distance threshold for visibility test",
    )
    parser.add_argument(
        "--thres_merge",
        type=int,
        default=20,
        help="thres to merge small isolated regions in the postprocess",
    )
    parser.add_argument(
        "--max_neighbor_distance",
        type=int,
        default=2,
        help="max logical distance for taking priimtive neighbors into account",
    )
    parser.add_argument(
        "--thres_trunc",
        type=float,
        default=0.0,
        help="trunc similarity that is under thres to 0",
    )
    parser.add_argument(
        "--seg_enhance",
        default=False,
        action="store_true",
        help="enhance the granularity of oversegmentation",
    )
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--vlm_type",
        type=str,
        default="clip",
        choices=["clip", "dinov3", "openseg"],
        help="which vlm type to use",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2,
        help="threshold factor for filtering features based on mean cosine similarity",
    )
    args = get_combined_args(parser)

    th_tuple = args.thres_connect.split(",")
    args.thres_connect = list(map(float, th_tuple))

    everything_seg(args)
