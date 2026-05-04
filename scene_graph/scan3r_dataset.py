# 3RScan dataset reader
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import pickle
import numpy as np
from collections import defaultdict
from transformers import AutoModel
from torch.utils.data import Dataset
import configs.scan3r.define as scan3rdefine
from configs.scan3r.util_metafile import read_json, read_txt_to_list
import torch
import torch.nn.functional as F
from scene_graph.network_utils import *

object_class_names = read_txt_to_list(os.path.join(
        scan3rdefine.SCAN3R_ROOT_PATH, "3DSSG_subset", "classes.txt"
    ))

relationship_class_names = read_txt_to_list(os.path.join(
    scan3rdefine.SCAN3R_ROOT_PATH, "3DSSG_subset", "relationships.txt"
))

spatial_synonyms = {
    "supported by": ["resting on", "held by", "carried by", "borne by", "mounted on", "placed on"],
    "left": ["to the left of", "on the left side of", "at left", "left-hand side", "leftward of"],
    "right": ["to the right of", "on the right side of", "at right", "right-hand side", "rightward of"],
    "front": ["in front of", "ahead of", "before", "facing toward", "anterior to"],
    "behind": ["at the back of", "in the rear of", "beyond", "posterior to"],
    "close by": ["near", "nearby", "adjacent to", "next to", "beside", "in proximity to"],
    "inside": ["within", "enclosed by", "contained in", "inside of", "in the interior of"],
    "bigger than": ["larger than", "greater than", "more massive than", "wider than", "taller than"],
    "smaller than": ["less than", "smaller in size", "narrower than", "shorter than", "lighter than"],
    "higher than": ["above", "over", "on top of", "elevated above", "situated above"],
    "lower than": ["below", "under", "beneath", "underneath", "at a lower level than"],
    "same symmetry as": ["mirrored with", "symmetric with", "having same axis", "aligned symmetrically with"],
    "same as": ["identical to", "equal to", "matching", "similar to", "equivalent to"],
    "attached to": ["connected to", "fixed to", "joined to", "bonded with", "fastened to"],
    "standing on": ["supported by", "resting on", "placed on top of", "mounted on", "set upon"],
    "lying on": ["resting on", "spread on", "placed flat on", "situated on"],
    "hanging on": ["suspended from", "dangling from", "attached from", "hooked on"],
    "connected to": ["linked with", "joined with", "attached to", "interfaced with", "coupled to"],
    "leaning against": ["supported against", "tilted toward", "resting against", "propped on"],
    "part of": ["component of", "subpart of", "included in", "member of", "contained in"],
    "belonging to": ["owned by", "associated with", "part of", "linked to"],
    "build in": ["embedded in", "integrated into", "incorporated in", "constructed within"],
    "standing in": ["located in", "positioned in", "situated within", "placed in"],
    "cover": ["covering", "overlaying", "protecting", "enveloping", "draped over"],
    "lying in": ["situated in", "resting in", "positioned within", "inside of"],
    "hanging in": ["suspended in", "floating in", "dangling in", "attached within"]
}

spatial_antonyms = {
    "none": ["connected to", "related to", "attached to"],
    "supported by": ["unsupported", "floating", "hanging from", "above without contact"],
    "left": ["right"],
    "right": ["left"],
    "front": ["behind", "back"],
    "behind": ["front", "ahead"],
    "close by": ["far from", "distant from", "away from"],
    "inside": ["outside", "external to", "surrounding"],
    "bigger than": ["smaller than", "less than", "narrower than"],
    "smaller than": ["bigger than", "larger than", "wider than"],
    "higher than": ["lower than", "below", "beneath"],
    "lower than": ["higher than", "above", "over"],
    "same symmetry as": ["asymmetric to", "different symmetry", "unaligned"],
    "same as": ["different from", "distinct from", "unequal to"],
    "attached to": ["detached from", "separated from", "disconnected from"],
    "standing on": ["standing on", "floating above", "lying on side", "unsupported"],
    "lying on": ["standing on", "hanging from", "upright"],
    "hanging on": ["standing on", "lying on", "supported from below"],
    "connected to": ["disconnected from", "isolated from", "separated from"],
    "leaning against": ["upright", "free standing", "unsupported"],
    "part of": ["separate from", "independent of", "not part of"],
    "belonging to": ["independent from", "not belonging to", "separate from"],
    "build in": ["detached from", "external to", "added on", "attached externally"],
    "standing in": ["outside of", "absent from", "out of"],
    "cover": ["uncover", "expose", "reveal", "open"],
    "lying in": ["outside of", "lying on", "above"],
    "hanging in": ["lying on", "standing on", "placed on surface"]
}

jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(torch.bfloat16).cuda()
jina_encode =  lambda x: jina_model.encode(x, task='text-matching', truncate_dim=512)
relationship_jina_feature = F.normalize(torch.from_numpy(jina_encode(relationship_class_names)), dim=1, p=2)
relationship_jina_dict = {relationship_class_names[i]:relationship_jina_feature[i] for i in range(len(relationship_class_names))}


all_words = set(relationship_class_names)
for main_word, syns in spatial_synonyms.items():
    all_words.add(main_word)
    all_words.update(syns)

# Encode all unique words in one pass
all_words = sorted(list(all_words))
all_word_features = torch.from_numpy(jina_encode(all_words))
all_word_features = F.normalize(all_word_features, dim=1, p=2)
jina_word_feat_dict = {w: all_word_features[i] for i, w in enumerate(all_words)}
jina_model = None  # free memory


def build_edge_gt_embeddings(edge_index, rel_gt, jina_dict, node_ids):
    """
    edge_index: (2, E)
    gt_relations: dict {(src, dst): [relation_text, ...]}
    jina_dict: dict {relation_text: torch.Tensor(512,)} — all normalized
    """
    edge_gt_labels_dict = defaultdict(list)
    for (a, b, cls_id, cls_name) in rel_gt:
        edge_gt_labels_dict[(a, b)].append(cls_name)


    E = edge_index.shape[1]
    gt_feat_dict = {}
    mask = torch.zeros(E, dtype=torch.bool)
    for e in range(E):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        key = (node_ids[src], node_ids[dst])
        if key in edge_gt_labels_dict:
            rel_feats = [jina_dict[r] for r in edge_gt_labels_dict[key] if r in jina_dict]
            if rel_feats:
                gt_feat_dict[e] = torch.stack(rel_feats)
                mask[e] = True
    #gt_feat = F.normalize(gt_feat, dim=-1)
    return gt_feat_dict, mask  # mask=True means we have ground truth

class Scan3RDataset(Dataset):
    def __init__(self, root, split="train", device="cuda", augment=False):
        split_file_path = os.path.join("3DSSG_subset", f'{split}_scans.txt')
        va_split_file_path = os.path.join("3DSSG_subset", f'validation_scans.txt')
        self.scan_list = read_txt_to_list(os.path.join(root, split_file_path)) + read_txt_to_list(os.path.join(root, va_split_file_path))
        self.root = root
        self.device = device
        self.augment = augment
        self.relationship_classes = relationship_class_names
        self.object_classes = object_class_names
        if split == "train":
            self.p_drop = 0.50
            self.p_synonym = 0.25
        else:
            self.p_drop = 0.75
            self.p_synonym = 0.75
        self.relationship_jina_feature = relationship_jina_feature.to(device)

    def __len__(self):
        return len(self.scan_list)
    
    def obj_gt_augmentation(self, points, labels, node_ids, R_thresh=np.pi/8, T_thresh=5):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # centering the point cloud
        centroid = np.mean(points, axis=0)
        points = points - centroid

        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
        
        # Random rotation angles within [-R_thresh, R_thresh]
        angles = np.random.uniform(-R_thresh, R_thresh, size=3)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        # Rotation matrix (Rx * Ry * Rz)
        Rx = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]])
        Ry = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]])
        #R = Rz @ Ry @ Rx
        R = Rz

        # Random translation within [-T_thresh, T_thresh]
        #t = np.random.uniform(-T_thresh, T_thresh, size=(3,))

        # Apply transformation
        points_aug = (R @ points.T).T #+ t
        

        object2obb = get_oriented_bounding_boxes(points_aug, labels, node_ids, remove_outlier=False)
        centers, Rs, extents, mass_centers = [], [], [], []
        for nid in node_ids:
            centers.append(object2obb[nid][0])
            Rs.append(object2obb[nid][1])
            extents.append(object2obb[nid][2])
            mass_centers.append(object2obb[nid][3])
        
        centers = np.stack(centers, axis=0)
        Rs = np.stack(Rs, axis=0)
        extents = np.stack(extents, axis=0)
        mass_centers = np.stack(mass_centers, axis=0)
        
        return [centers, Rs, extents], mass_centers, points_aug
    
    def rel_gt_augmentation(sel, rel_gt, p_drop=0.5, p_synonym=0.6):
        """
        Randomly drop some relationships for data augmentation.
        rel_gt: list of (src, dst, rel_id, rel_name)
        p: drop probability
        return: augmented rel_gt
        """
        drop_aug_rel_gt = []
        for (a, b, cls_id, cls_name) in rel_gt:
            if torch.rand(1).item() > p_drop:
                drop_aug_rel_gt.append((a, b, cls_id, cls_name))
        
        noisy_rel_gt = []
        for (a, b, cls_id, cls_name) in drop_aug_rel_gt:
            if torch.rand(1).item() < p_synonym:
                synonyms = spatial_synonyms[cls_name]
                torch.randperm(len(synonyms))
                new_name = synonyms[0]
                noisy_rel_gt.append((a, b, cls_id, new_name))
            else:
                noisy_rel_gt.append((a, b, cls_id, cls_name))

        return noisy_rel_gt
    
    def mix_edge_labels_to_jina_feature(self, rels, jina_word_feat_dict):
        """
        rels: list of (a,b,cls_id,cls_name)
        jina_word_feat_dict: precomputed {word: 512-d tensor}
        """
        rel_dict = defaultdict(list)
        for (a, b, cls_id, cls_name) in rels:
            rel_dict[(a, b)].append(cls_name)

        edge_jina_feat_dict = {}
        for key, names in rel_dict.items():
            feats = []
            for name in names:
                # if synonym available, average all of its synonyms as well
                if name in jina_word_feat_dict:
                    feats.append(jina_word_feat_dict[name])
                elif name in spatial_synonyms:
                    for syn in spatial_synonyms[name]:
                        if syn in jina_word_feat_dict:
                            feats.append(jina_word_feat_dict[syn])
            if feats:
                edge_jina_feat_dict[key] = F.normalize(torch.stack(feats).mean(dim=0, keepdim=True), dim=1, p=2)
            else:
                # fallback to zero or random small noise
                edge_jina_feat_dict[key] = torch.zeros(1, 512, device=list(jina_word_feat_dict.values())[0].device)
        return edge_jina_feat_dict
    
    def assign_edge_features(self, edge_index, rel_feat_dict, node_ids, feat_dim=512, device=None):
        """
        Assign edge features based on (src, dst) → feature dict.
        Unmatched edges will have zero features.

        Args:
            edge_index (torch.Tensor): (2, E) tensor of edge indices.
            rel_feat_dict (dict): {(src, dst): torch.Tensor(1, feat_dim)}.
            feat_dim (int): Dimension of each feature vector.
            device (torch.device, optional): Target device for output.

        Returns:
            torch.Tensor: (E, feat_dim) edge feature tensor.
        """
        E = edge_index.shape[1]
        device = device or edge_index.device
        edge_feat = torch.zeros((E, feat_dim), device=device)

        # Build a lookup mapping for fast access
        for e_idx in range(E):
            src = int(edge_index[0, e_idx])
            dst = int(edge_index[1, e_idx])
            key = (node_ids[src], node_ids[dst])
            if key in rel_feat_dict:
                feat = rel_feat_dict[key].to(device)
                edge_feat[e_idx] = feat.squeeze(0)

        return edge_feat

    def impute_edge_features(self, edge_index, edge_feat):
        """
        Fill zero-edge features with mean of neighboring edges.
        edge_feat: (E, D)
        """
        E, D = edge_feat.shape
        src, dst = edge_index

        # Build adjacency: node → list of edge indices
        node2edges = defaultdict(list)
        for e in range(E):
            node2edges[int(src[e])].append(e)
            node2edges[int(dst[e])].append(e)

        edge_feat_new = edge_feat.clone()
        for e in range(E):
            if torch.allclose(edge_feat[e], torch.zeros(D, device=edge_feat.device)):
                neighbors = node2edges[int(src[e])] + node2edges[int(dst[e])]
                neighbors = list(set(neighbors))
                neighbor_feats = edge_feat[neighbors]
                mask = (neighbor_feats.abs().sum(dim=1) != 0)
                if mask.any():
                    edge_feat_new[e] = neighbor_feats[mask].mean(dim=0)
        return edge_feat_new



    def __getitem__(self, idx):
        scan_pickle_path = os.path.join(self.root, "packed_data_openseg", f"{self.scan_list[idx]}.pkl")
        with open(scan_pickle_path, "rb") as f:
            sample = pickle.load(f)
        
        
        node_id = sample['node_id']
        rel_gt = sample['rel_gt']
        node_gt = sample['node_gt']
        node_clip_feat = sample['node_clip_feat']
        edge_index = torch.as_tensor(sample['neighbor_edge_index'], dtype=torch.long).t() # (2, E)
        point_label = sample['labels']

        #print(edge_index.shape)
        
        rel_gt_jina_feat_dict, rel_gt_edge_index_mask = build_edge_gt_embeddings(edge_index, rel_gt, relationship_jina_dict, node_id)

        if self.augment:
            [centers, Rs, extents], mass_centers, points = self.obj_gt_augmentation(
                sample['points'], point_label, node_id, R_thresh=np.pi/4, T_thresh=5)
            node_geo_feat = build_geom_features(centers, Rs, extents)
            noisy_rel_gt = self.rel_gt_augmentation(rel_gt, p_drop=self.p_drop, p_synonym=self.p_synonym)
            noisy_rel_gt_jina_feat_dict = self.mix_edge_labels_to_jina_feature(noisy_rel_gt, jina_word_feat_dict)
            edge_init_feat = self.assign_edge_features(edge_index, noisy_rel_gt_jina_feat_dict, node_id, feat_dim=512, device="cpu")
            
        else:
            centers, Rs, extents = sample['obbs']
            mass_centers = sample['mass centers']
            points = sample['points']
            node_geo_feat = sample['node_geo_feat']
            edge_init_feat = torch.zeros((edge_index.shape[1], 512), dtype=torch.float32)
        
        edge_geo_feat = get_edge_geom_features(centers, Rs, extents, edge_index)
        
        data = {}
        data['scan_id'] = self.scan_list[idx]
        data['point_label'] = torch.as_tensor(point_label, dtype=torch.long)
        data['points'] = torch.as_tensor(points, dtype=torch.float32)
        data['edge_index'] = edge_index
        data['node_geo_feat'] = node_geo_feat
        data['edge_geo_feat'] = edge_geo_feat
        data['node_clip_feat'] = node_clip_feat
        data['edge_init_feat'] = edge_init_feat
        data['rel_gt'] = rel_gt
        data['rel_gt_jina_feat_dict'] = rel_gt_jina_feat_dict
        data['rel_gt_edge_index_mask'] = rel_gt_edge_index_mask
        data['node_id'] = node_id
        data['node_gt'] = node_gt
        
        return data
    
    def collate_fn(self, batch):
        batch_out = {}

        scan_ids = []
        node_geo_feats, node_clip_feats = [], []
        edge_geo_feats, edge_init_feats = [], []
        points_list, point_labels_list = [], []
        rel_gt_list, rel_gt_jina_list, rel_gt_mask_list = [], [], []
        node_id_list, node_gt_list = [], []
        edge_index_list = []

        node_offset = 0
        for data in batch:
            scan_ids.append(data['scan_id'])
            node_geo_feats.append(data['node_geo_feat'])
            node_clip_feats.append(data['node_clip_feat'])
            edge_geo_feats.append(data['edge_geo_feat'])
            edge_init_feats.append(data['edge_init_feat'])
            points_list.append(data['points'])
            point_labels_list.append(data['point_label'])

            rel_gt_list.append(data['rel_gt'])
            rel_gt_jina_list.append(data['rel_gt_jina_feat_dict'])
            rel_gt_mask_list.append(data['rel_gt_edge_index_mask'])
            node_id_list.append(data['node_id'])
            node_gt_list.append(data['node_gt'])

            edge_index = data.get('edge_index', None)
            if edge_index is None:
                raise KeyError(f"'edge_index' missing for scan {data['scan_id']}")
            edge_index_list.append(edge_index + node_offset)
            node_offset += data['node_geo_feat'].shape[0]

        device = node_geo_feats[0].device

        batch_out['scan_id'] = scan_ids
        batch_out['node_geo_feat'] = torch.cat(node_geo_feats, dim=0).to(device)
        batch_out['node_clip_feat'] = torch.cat(node_clip_feats, dim=0).to(device)
        batch_out['edge_geo_feat'] = torch.cat(edge_geo_feats, dim=0).to(device)
        batch_out['edge_init_feat'] = torch.cat(edge_init_feats, dim=0).to(device)
        batch_out['points'] = torch.cat(points_list, dim=0).to(device)
        batch_out['point_label'] = torch.cat(point_labels_list, dim=0).to(device)
        batch_out['edge_index'] = torch.cat(edge_index_list, dim=1).long().to(device)

        batch_out['rel_gt'] = rel_gt_list
        batch_out['rel_gt_jina_feat_dict'] = rel_gt_jina_list
        batch_out['rel_gt_edge_index_mask'] = rel_gt_mask_list
        batch_out['node_id'] = node_id_list
        batch_out['node_gt'] = node_gt_list

        batch_out['batch_node'] = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.long, device=device)
            for i, x in enumerate(node_geo_feats)
        ])
        batch_out['batch_edge'] = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.long, device=device)
            for i, x in enumerate(edge_geo_feats)
        ])

        batch_out['num_nodes_per_graph'] = [x.shape[0] for x in node_geo_feats]
        batch_out['num_edges_per_graph'] = [x.shape[0] for x in edge_geo_feats]

        return batch_out