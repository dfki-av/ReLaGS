"""
This module defines the SemanticNAG class, which constructs a hierarchical superpoint graph with semantic features.
It builds upon the Nested Adjacency Graph (NAG) structure from the SPT (Superpoint Transformer) library (https://arxiv.org/abs/2306.08045).
"""

import torch
from typing import List, Tuple
import sys

sys.path.append("ext/")
from spt.data import Data, NAG, Cluster


class SemanticNAG:
    def __init__(self, labels: List[torch.Tensor], feat: torch.Tensor):
        labels = [label.cuda() for label in labels]
        self.labels = labels  # List of 1D tensors, one per hierarchy level
        self.nag = self.build_nag_from_multilevel_labels(labels)
        self.feat = feat
        self.gaussian_num = labels[0].shape[0]

    def _get_root_candidates(self, sim_at_levels, filter_small):
        """Find top root-level candidates filtered by minimum size."""
        root_level = len(self.labels) - 1
        root_sim_val, root_candidate_ids = torch.topk(
            sim_at_levels[-1], min(10, sim_at_levels[-1].shape[0])
        )
        too_small_mask = [
            ((self.labels[root_level] == i).sum() > filter_small).item()
            for i in root_candidate_ids
        ]
        return root_sim_val[too_small_mask], root_candidate_ids[too_small_mask]

    def _build_root_candidates_with_leaf_info(
        self, root_candidate_ids, root_sim_val, sim_at_levels, leaf_level
    ):
        """Build root-level candidates enriched with leaf-level similarity data."""
        root_level = len(self.labels) - 1
        candidates = []
        for i, root_id in enumerate(root_candidate_ids):
            root_id = root_id.item()
            root_mask = self.labels[-1] == root_id
            inmask_leaf = self.labels[leaf_level][root_mask]
            inmask_leaf_indices = torch.unique(inmask_leaf).cpu().numpy()
            leaf_sim = sim_at_levels[leaf_level - 1][inmask_leaf_indices]
            candidates.append(
                [
                    root_level,
                    root_id,
                    root_id,
                    root_sim_val[i].item(),
                    inmask_leaf_indices,
                    leaf_sim,
                    (self.labels[root_level] == root_id).sum().item(),
                ]
            )
        candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
        return candidates

    def _analyze_leaf_candidates(self, root_candidates, leaf_level):
        """Analyze leaf-level candidates and promote those with higher similarity than root."""
        overall_candidates = []
        for i, tup in enumerate(root_candidates):
            _, root_id, _, root_sim_val, leaf_indices, leaf_sim, _ = tup
            leaf_node_pt_nums = [
                (self.labels[leaf_level] == leaf_id).sum().item()
                for leaf_id in root_candidates[i][4]
            ]
            too_small_mask = [
                pt_num >= (sum(leaf_node_pt_nums) * 0.01)
                for pt_num in leaf_node_pt_nums
            ]
            if len(leaf_indices) == 1:
                overall_candidates.append(tup)
            elif torch.all(leaf_sim[too_small_mask] < root_sim_val):
                overall_candidates.append(tup)
            elif torch.any(leaf_sim > root_sim_val):
                leaf_sim = leaf_sim[too_small_mask]
                leaf_indices = leaf_indices[too_small_mask]
                leaf_candidate_mask = (
                    torch.where(leaf_sim > root_sim_val)[0].cpu().numpy()
                )
                leaf_sim_vals = leaf_sim[leaf_candidate_mask]
                leaf_candidate_ids = leaf_indices[leaf_candidate_mask]
                overall_candidates.extend(
                    [
                        (
                            leaf_level,
                            root_id,
                            leaf_candidate_ids[j],
                            leaf_sim_vals[j].item(),
                            None,
                            None,
                            (self.labels[leaf_level] == leaf_candidate_ids[j])
                            .sum()
                            .item(),
                        )
                        for j in range(len(leaf_candidate_ids))
                    ]
                )
        return overall_candidates

    def _filter_by_similarity_gap(self, candidates, remove_small):
        """Filter candidates based on similarity gap analysis."""
        candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
        sim_deltas = [0] + [
            candidates[i][3] - candidates[i + 1][3] for i in range(len(candidates) - 1)
        ]
        sim_delta_max_idx = sim_deltas.index(max(sim_deltas)) if sim_deltas else 0

        for i in range(len(candidates)):
            if i == sim_delta_max_idx:
                print("----- Selected candidates above -----")
            level, root_index, index, sim_val, leaf_indices, leaf_sim, num_pts = (
                candidates[i]
            )
            print(
                f"Level: {level}, Root ID: {root_index}, Index: {index}, "
                f"Sim: {sim_val:.4f}, Num_pts: {num_pts}"
            )
        if remove_small:
            return candidates[:sim_delta_max_idx]
        return candidates

    def _get_related_gaussians(self, candidates):
        """Build binary Gaussian mask from candidate superpoints."""
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for tup in candidates:
            level, _, index, _, _, _, _ = tup
            lowest_idx = torch.where(self.labels[level] == index)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians

    def search_matched_superpoint_in_mhtree(
        self,
        sim_at_levels: List[torch.Tensor],
        topk: int = 5,
        level_until: int = -1,
        filter_small: int = 50,
        remove_small: bool = True,
    ) -> Tuple[torch.Tensor, List[list]]:
        assert len(sim_at_levels) == len(self.labels) - 1, (
            "Number of similarity matrices must match number of levels"
        )

        root_level = len(self.labels) - 1
        leaf_level = root_level - 1

        root_sim_val, root_candidate_ids = self._get_root_candidates(
            sim_at_levels, filter_small
        )

        if level_until == -1:
            overall_candidates = [
                [
                    root_level,
                    root_candidate_ids[i].item(),
                    root_candidate_ids[i].item(),
                    round(root_sim_val[i].item(), 3),
                    None,
                    None,
                    (self.labels[root_level] == root_candidate_ids[i].item())
                    .sum()
                    .item(),
                ]
                for i in range(len(root_candidate_ids))
            ]
            overall_candidates = sorted(
                overall_candidates, key=lambda x: x[3], reverse=True
            )
            overall_candidates = overall_candidates[:topk]
        else:
            root_level_candidates = self._build_root_candidates_with_leaf_info(
                root_candidate_ids, root_sim_val, sim_at_levels, leaf_level
            )
            root_level_candidates = root_level_candidates[:topk]
            overall_candidates = self._analyze_leaf_candidates(
                root_level_candidates, leaf_level
            )
            overall_candidates = self._filter_by_similarity_gap(
                overall_candidates, remove_small
            )

        if not overall_candidates:
            return torch.zeros(self.gaussian_num, 1, dtype=torch.float32), []

        rel_gaussians = self._get_related_gaussians(overall_candidates)
        return rel_gaussians, overall_candidates

    @staticmethod
    def build_nag_from_multilevel_labels(labels: List[torch.Tensor]) -> NAG:
        """Build a Nested Adjacency Graph (NAG) from multi-level labels.

        Args:
            labels: List of Tensors, each shape (N,), mapping points to cluster IDs at each level.
                e.g. [label_lvl0, label_lvl1, label_lvl2]

        Returns:
            NAG: Nested Adjacency Graph structure
        """
        assert len(labels) >= 2, "At least two levels required to construct NAG"
        device = labels[0].device
        N = labels[0].shape[0]
        num_levels = len(labels)

        data_list = [Data(num_nodes=N, super_index=labels[0])]
        prev_sub = None

        for i in range(num_levels):
            data = Data()
            # Compute sub for each level and super_index from the previous level
            if i == 0:
                upper_labels = labels[i]
                lower_labels = torch.arange(N, device=device)
                sorted_upper_labels, perm = torch.sort(upper_labels)
                sorted_lower_labels = lower_labels[perm]
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(
                    sorted_upper_labels, minlength=num_clusters
                )
                pointers = torch.zeros(
                    num_clusters + 1, dtype=torch.long, device=device
                )
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                prev_sub = Cluster(
                    pointers=pointers, points=sorted_lower_labels, dense=False
                )
            else:
                data = Data(num_nodes=labels[i - 1].max().item() + 1)
                data.sub = prev_sub
                upper_labels = labels[i].long()
                lower_labels = labels[i - 1].long()

                # 1. remove duplicates based on lower labels
                unique_lower_labels, inv = torch.unique(
                    lower_labels, sorted=True, return_inverse=True
                )
                unique_upper_labels = torch.zeros_like(unique_lower_labels)
                unique_upper_labels[inv] = upper_labels

                data.super_index = unique_upper_labels
                # 2. sort based on upper labels, construct next sub
                sorted_upper_labels, perm = torch.sort(unique_upper_labels)
                sorted_lower_labels = unique_lower_labels[perm]
                # upper labels reflect the changes from lower level
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(
                    sorted_upper_labels, minlength=num_clusters
                )
                pointers = torch.zeros(
                    num_clusters + 1, dtype=torch.long, device=device
                )
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                prev_sub = Cluster(
                    pointers=pointers, points=sorted_lower_labels, dense=False
                )
                data_list.append(data)
        data_list.append(Data(sub=prev_sub, num_nodes=labels[-1].max().item() + 1))
        return NAG(data_list)
