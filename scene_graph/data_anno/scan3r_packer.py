# 3RScan dataset reader
from argparse import ArgumentParser
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pickle
import numpy as np
from torch.utils.data import Dataset
import configs.scan3r.define as scan3rdefine
from configs.scan3r.util_metafile import read_json
from configs.scan3r.util_label import getLabelMapping
import trimesh
import open_clip
from pycocotools import mask as mask_utils
from PIL import Image
import torch
import torch.nn.functional as F
import open3d as o3d
import tensorflow as tf
from collections import defaultdict
from scene_graph.network_utils import (
    aggregate_clip,
    build_geom_features,
    obb_min_distance_matrix_numpy,
)
import cv2

def default_object_entry():
    return {"feats": [], "pixel_nums": []}

def read_bytes(path):
    """Read bytes for OpenSeg model running."""
    # Modern TF file I/O works directly via tf.io.gfile
    with tf.io.gfile.GFile(path, 'rb') as f:
        return f.read()

def extract_openseg_img_feature(img_dir, openseg_model, img_size=None, regional_pool=True, r3scan=False):
    """Extract per-pixel OpenSeg features."""

    # Modern TensorFlow ops (no compat.v1)
    text_emb = tf.zeros([1, 1, 768], dtype=tf.float32)

    # load RGB image
    if r3scan:
        if isinstance(img_dir, str):
            image = Image.open(img_dir).rotate(-90, expand=True)
        elif isinstance(img_dir, Image.Image):
            image = img_dir.rotate(-90, expand=True)
        np_image_string = tf.io.encode_jpeg(np.asarray(image))
    else:
        np_image_string = read_bytes(img_dir)

    # Run OpenSeg model in TF2 eager mode
    results = openseg_model.signatures['serving_default'](
        inp_image_bytes=tf.convert_to_tensor(np_image_string),
        inp_text_emb=text_emb
    )

    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]

    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]

    if "r3scan" in str(img_dir):
        image_embedding_feat = tf.image.rot90(image_embedding_feat, k=1)

    # Modern resize op (align_corners deprecated in resize_nearest_neighbor)
    if img_size is not None:
        feat_2d = tf.cast(
            tf.image.resize(
                image_embedding_feat, img_size, method="nearest"
            )[0],
            dtype=tf.float16
        ).numpy()
    else:
        feat_2d = tf.cast(image_embedding_feat[0], dtype=tf.float16).numpy()

    del results
    del image_embedding_feat

    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1)
    return feat_2d


class Scan3RDataset_packer(Dataset):
    def __init__(
        self,
        scan_list,
        root,
        pkl_root,
        data_root=scan3rdefine.SCAN3R_ROOT_PATH,
        vlm_type="clip",
        split="train",
        openseg_model_path=None,
        clip_batch_size=128,
        device=None,
    ):
        self.scan_list = scan_list
        self.data_root = data_root
        self.pkl_root = pkl_root
        self.clip_batch_size = clip_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names, self.label_name_mapping, self.label_id_mapping = getLabelMapping("3rscan160")
        self.scene_data, _ = read_json(root, split)
        if vlm_type == "clip":
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                precision="fp16" if self.device == "cuda" else "fp32",
            )
            self.clip_model.eval().to(self.device)
        elif vlm_type == "openseg":
            if openseg_model_path is None:
                raise ValueError("--openseg_model_path is required when --vlm_type openseg")
            self.openseg_model = tf.saved_model.load(openseg_model_path, tags=[tf.saved_model.SERVING])
        else:
            raise ValueError(f"Unsupported vlm_type: {vlm_type}")
        self.vlm_type = vlm_type

    def __len__(self):
        return len(self.scan_list)

    def _load_mesh(self, scan_id):
        scene_pth = os.path.join(self.data_root, scan_id)
        label_mesh_pth = os.path.join(scene_pth, scan3rdefine.LABEL_FILE_NAME)
        return trimesh.load(label_mesh_pth, process=False)

    def _clean_no_image_nodes(self, node_data, edge_data, object2image):
        cleaned_node_data = {}
        valid_obj_ids = set()
        for obj_id in node_data:
            if obj_id in object2image and len(object2image[obj_id]) > 0:
                cleaned_node_data[int(obj_id)] = node_data[obj_id]
                valid_obj_ids.add(int(obj_id))

        cleaned_edge_data = []
        for rel in edge_data:
            subj_id, obj_id, predicate, _ = rel
            if subj_id in valid_obj_ids and obj_id in valid_obj_ids:
                cleaned_edge_data.append(rel)

        return cleaned_node_data, cleaned_edge_data

    def _load_per_scene_rgb_images(self, scan_id, image_names=None):
        rgb_dict = {}
        seq_root = os.path.join(self.data_root, scan_id, "sequence")
        if image_names is None:
            image_files = sorted([f for f in os.listdir(seq_root) if f.endswith('.jpg')])
            image_names = image_files[::10]
        for img_file in sorted(image_names):
            img_path = os.path.join(seq_root, img_file)
            rgb_dict[img_file] = Image.open(img_path).convert("RGB")

        return rgb_dict

    def obb_min_distance_matrix_numpy(self, centers, rotations, extents, dist_thresh=None, topk=None):
        """
        Compute approximate pairwise minimal distances between oriented bounding boxes (OBBs)
        and return both the distance matrix and adjacency mask.

        Args:
            centers:   (N, 3) ndarray - OBB centers
            rotations: (N, 3, 3) ndarray - rotation matrices (columns are axes)
            extents:   (N, 3) ndarray - half-lengths along local x/y/z axes
            dist_thresh: float, optional. If given, adjacency_mask[i,j] = True if distance < thresh
            topk: int, optional. If given, keep top-K nearest neighbors per node (including both directions)
                    (overrides dist_thresh if both given)

        Returns:
            dist_mat:        (N, N) ndarray of minimal OBB distances (symmetric, zeros on diagonal)
            adjacency_mask:  (N, N) boolean ndarray (True where edge should exist)
        """
        return obb_min_distance_matrix_numpy(centers, rotations, extents, dist_thresh, topk)

    def _mass_center_distance_matrix_numpy(self, mass_centers, dist_thresh=None, topk=None):
        N = len(mass_centers)
        mass_centers = np.asarray(mass_centers, float)
        d = mass_centers[:, None, :] - mass_centers[None, :, :]  # (N, N, 3)
        dist = np.linalg.norm(d, axis=-1)                          # (N, N)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)

        # --- adjacency mask ---
        if topk is not None:
            # Keep top-K smallest distances (excluding self)
            adj = np.zeros_like(dist, dtype=bool)
            idx = np.argsort(dist, axis=1)
            for i in range(N):
                k = min(topk + 1, N)
                adj[i, idx[i, 1:k]] = True  # exclude self
            adjacency_mask = np.logical_or(adj, adj.T)  # make symmetric
        elif dist_thresh is not None:
            adjacency_mask = dist < dist_thresh
            np.fill_diagonal(adjacency_mask, False)
        else:
            adjacency_mask = np.ones((N, N), dtype=bool)
            np.fill_diagonal(adjacency_mask, False)

        return dist, adjacency_mask

    def _gather_object_images_and_clip_features(self, object2image, rgb_dict, node_ids):
        cropped_image_mask_batch = []
        pixel_nums = []
        obj_id_map = []
        # Gathering all masked and croppred images for each object in the scene
        for obj_id in node_ids:
            obj_img_data = object2image[str(obj_id)]
            for i in range(len(obj_img_data)):
                image_name, pixel_num, ratio, bbox, mask_rle = obj_img_data[i]
                decoded_mask = mask_utils.decode(mask_rle)
                x1, y1, x2, y2 = bbox
                image = rgb_dict[image_name]
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_mask = decoded_mask[y1:y2, x1:x2]
                masked_cropped_image = Image.fromarray(np.array(cropped_image) * np.expand_dims(cropped_mask, axis=-1))
                padded_img = self._pad_img(np.array(masked_cropped_image))
                resized_img = Image.fromarray(padded_img).resize((224, 224))
                cropped_image_mask_batch.append(resized_img)
                pixel_nums.append(pixel_num)
                obj_id_map.append(int(obj_id))
        if not cropped_image_mask_batch:
            return {}
        pixel_nums = np.array(pixel_nums)

        # Batched process to get CLIP features
        cropped_image_mask_batch = np.stack(cropped_image_mask_batch, axis=0)
        cropped_image_mask_batch = torch.as_tensor(cropped_image_mask_batch).permute(0, 3, 1, 2) / 255.0
        if self.device == "cuda":
            cropped_image_mask_batch = cropped_image_mask_batch.half()
        else:
            cropped_image_mask_batch = cropped_image_mask_batch.float()
        clip_feature_chunks = []
        with torch.no_grad():
            for start in range(0, cropped_image_mask_batch.shape[0], self.clip_batch_size):
                batch = cropped_image_mask_batch[start:start + self.clip_batch_size].to(self.device)
                clip_features = self.clip_model.encode_image(batch)
                clip_features = F.normalize(clip_features, dim=1, p=2)
                clip_feature_chunks.append(clip_features.cpu())
        clip_features = torch.cat(clip_feature_chunks, dim=0)

        object2clip = {}
        for obj_id in node_ids:
            obj_id_mask = np.asarray([obj_id == oid for oid in obj_id_map], dtype=bool)
            #print(obj_id, obj_id_mask, obj_id_map, sum(obj_id_mask))
            if obj_id_mask.sum() > 0:
                object2clip[obj_id] = {"feats": clip_features[obj_id_mask], "pixel_nums": pixel_nums[obj_id_mask]}
                #print(obj_id, object2clip[obj_id]["pixel_nums"], object2clip[obj_id]["feats"].shape)
            #else:
                #object2clip[int(obj_id)] = torch.empty((0, clip_features.shape[1])).cuda()
        return object2clip

    def _gather_object_images_and_openseg_features(self, object2image, image_dict, node_ids, model):
        image2object = defaultdict(dict)
        for obj_id in node_ids:
            obj_img_data = object2image[str(obj_id)]
            for i in range(len(obj_img_data)):
                image_name, pixel_num, ratio, bbox, rle = obj_img_data[i]
                image2object[image_name][obj_id] = {
                                                    "rle": rle,
                                                    "pixel_num": pixel_num,
                                                    "ratio": ratio,
                                                    "bbox": bbox,
                                                }
        object2clip = {}
        for image_name, obj_entries in image2object.items():
            if image_name not in image_dict:
                continue
            image = image_dict[image_name]
            H_r, W_r = image.size[1], image.size[0]
            descriptors = extract_openseg_img_feature(image, model, img_size=[int(H_r/2), int(W_r/2)], r3scan=True)
            descriptors = torch.rot90(descriptors, k=1, dims=[1,2])

            for inst, obj_data in obj_entries.items():
                if inst not in object2clip:
                    object2clip[inst] = {"feats": [], "pixel_nums": []}
                mask_rle = obj_data["rle"]
                pixel_num = obj_data["pixel_num"]
                decoded_mask = mask_utils.decode(mask_rle)
                resized_mask = cv2.resize(decoded_mask, (descriptors.shape[2], descriptors.shape[1]), interpolation=cv2.INTER_NEAREST)
                obj_mask = torch.from_numpy(resized_mask).bool()
                if obj_mask.sum() == 0:
                    continue
                obj_descriptors = descriptors[:, obj_mask].T  # (N_pixels, feat_dim)
                obj_feat = obj_descriptors.mean(dim=0)  # (feat_dim,)
                if torch.isnan(obj_feat).sum() > 0 or torch.isinf(obj_feat).sum() > 0:
                    print("Warning: NaN or Inf in obj_feat!")
                object2clip[inst]["feats"].append(obj_feat)
                object2clip[inst]["pixel_nums"].append(pixel_num)

        # stack features into tensors
        for inst in object2clip:
            if not object2clip[inst]["feats"]:
                continue
            object2clip[inst]["feats"] = torch.stack(object2clip[inst]["feats"], dim=0)
            object2clip[inst]["pixel_nums"] = np.asarray(object2clip[inst]["pixel_nums"])

        return {
            inst: data
            for inst, data in object2clip.items()
            if isinstance(data["feats"], torch.Tensor) and data["feats"].shape[0] > 0
        }

    def _pad_img(self, img):
        h, w, _ = img.shape
        l = max(w,h)
        pad = np.zeros((l,l,3), dtype=np.uint8)
        if h > w:
            pad[:,(h-w)//2:(h-w)//2 + w, :] = img
        else:
            pad[(w-h)//2:(w-h)//2 + h, :, :] = img
        return pad

    def _get_oriented_bounding_boxes(self, points, labels, obj_ids):
        obj_id_to_obb = {}
        for obj_id in obj_ids:
            obj_points = points[labels.squeeze() == obj_id]
            # --- Convert numpy array to Open3D PointCloud ---
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            mass_center = np.asarray(pcd.get_center())

            # --- Compute oriented bounding box ---
            obb = pcd.get_oriented_bounding_box()

            # --- Extract as numpy arrays ---
            center = np.asarray(obb.center)
            R = np.asarray(obb.R)
            extent = np.asarray(obb.extent)
            obj_id_to_obb[obj_id] = (center, R, extent, mass_center)
        return obj_id_to_obb

    def _get_per_object_pointclouds(self, points, labels):
        obj_ids = np.unique(labels)
        obj_id_to_pcd = {}
        for obj_id in obj_ids:
            if obj_id == 0:
                continue
            obj_points = points[labels.squeeze() == obj_id]
            if obj_points.shape[0] < 5:
                continue
            obj_id_to_pcd[obj_id] = obj_points
        return obj_id_to_pcd

    def __getitem__(self, idx):
        scan_id = self.scan_list[idx]
        scene_graph = self.scene_data[scan_id]
        node_data = scene_graph['obj']
        edge_data = scene_graph['rel']
        pickle_path = os.path.join(self.pkl_root, f"{scan_id}_object2image.pkl")
        with open(pickle_path, 'rb') as f:
            object2image = pickle.load(f)

        label_mesh = self._load_mesh(scan_id)
        points = np.array(label_mesh.vertices)
        labels = np.array(label_mesh.metadata['_ply_raw']['vertex']['data']['objectId']).astype(np.int32)

        node_data, edge_data = self._clean_no_image_nodes(node_data, edge_data, object2image)
        node_ids = sorted(list(node_data.keys()))
        required_image_names = {
            entry[0]
            for obj_id in node_ids
            for entry in object2image[str(obj_id)]
        }
        rgb_dict = self._load_per_scene_rgb_images(scan_id, required_image_names)
        if self.vlm_type == "clip":
            object2clip = self._gather_object_images_and_clip_features(object2image, rgb_dict, node_ids)
        elif self.vlm_type == "openseg":
            object2clip = self._gather_object_images_and_openseg_features(object2image, rgb_dict, node_ids, self.openseg_model)

        node_ids = [nid for nid in node_ids if nid in object2clip]
        node_data = {nid: node_data[nid] for nid in node_ids}
        valid_obj_ids = set(node_ids)
        edge_data = [rel for rel in edge_data if rel[0] in valid_obj_ids and rel[1] in valid_obj_ids]
        if not node_ids:
            raise ValueError(f"No valid object features extracted for scan {scan_id}")

        object2obb = self._get_oriented_bounding_boxes(points, labels, node_ids)

        centers, Rs, extents, mass_centers, clip_feat = [], [], [], [], []
        for nid in node_ids:
            centers.append(object2obb[nid][0])
            Rs.append(object2obb[nid][1])
            extents.append(object2obb[nid][2])
            mass_centers.append(object2obb[nid][3])
            clip_feat.append(aggregate_clip(object2clip[nid]["feats"], object2clip[nid]["pixel_nums"]))

        centers = np.stack(centers, axis=0)
        Rs = np.stack(Rs, axis=0)
        extents = np.stack(extents, axis=0)
        mass_centers = np.stack(mass_centers, axis=0)
        dist, adj_mask = self.obb_min_distance_matrix_numpy(centers, Rs, extents, dist_thresh=0.2)
        mass_dist, mass_adj_mask = self._mass_center_distance_matrix_numpy(mass_centers, dist_thresh=0.5)

        node_geom_feat = build_geom_features(centers, Rs, extents)
        clip_feat = torch.stack(clip_feat, dim=0)

        # adj_mask to non zeros edge index
        edge_index = np.nonzero(adj_mask)
        edge_index = [[edge_index[0][i], edge_index[1][i]] for i in range(len(edge_index[0]))]

        return {"node_geo_feat": node_geom_feat,
                "node_clip_feat": clip_feat,
                "neighbor_edge_index": edge_index,
                "node_id": node_ids,
                "node_gt": node_data,
                "rel_gt": edge_data,
                "original_clip_node_dict": object2clip,
                "points": points,
                "labels": labels,
                "obb_dist": dist,
                "mass_center_dist": mass_dist,
                "obbs": [centers, Rs, extents],
                "mass centers": mass_centers
                }

def extract_object_pcd(points, labels, obj_id):
    obj_points = points[labels.squeeze() == obj_id]
    if obj_points.shape[0] < 5:
        return None
    # --- Convert numpy array to Open3D PointCloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    return pcd


if __name__ == "__main__":
    parser = ArgumentParser(description="Pack 3RScan object-image annotations into graph training pickles")
    parser.add_argument("--root", default=None, type=str, help="path of 3DSSG dataset")
    parser.add_argument("--data_root", default=scan3rdefine.SCAN3R_ROOT_PATH, type=str, help="path of raw 3RScan scans")
    parser.add_argument("--pkl_root", default=None, type=str, help="folder containing *_object2image.pkl files")
    parser.add_argument("--output_dir", default=None, type=str, help="folder to save packed graph pickles")
    parser.add_argument("--split", default="train", type=str, help="dataset split to process")
    parser.add_argument("--vlm_type", default="openseg", choices=["clip", "openseg"], help="visual feature backend")
    parser.add_argument("--openseg_model_path", default=None, type=str, help="path to exported OpenSeg SavedModel")
    parser.add_argument("--clip_batch_size", default=128, type=int, help="CLIP crop batch size")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing packed pickles")
    args = parser.parse_args(sys.argv[1:])

    root = args.root or os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, "3DSSG_subset")
    pkl_root = args.pkl_root or os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, "scan3r_obj2frame")
    default_output_name = f"packed_data_{args.vlm_type}"
    output_dir = args.output_dir or os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, default_output_name)
    os.makedirs(output_dir, exist_ok=True)

    scene_data, selected_scans = read_json(root, args.split)
    testing_list = sorted([scan for scan in selected_scans if scan in scene_data])
    dataset = Scan3RDataset_packer(
        scan_list=testing_list,
        root=root,
        pkl_root=pkl_root,
        data_root=args.data_root,
        vlm_type=args.vlm_type,
        split=args.split,
        openseg_model_path=args.openseg_model_path,
        clip_batch_size=args.clip_batch_size,
    )
    for i in range(0, len(dataset)):
        save_path = os.path.join(output_dir, f"{testing_list[i]}.pkl")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Packed pickle for {testing_list[i]} already exists, skipping...")
            continue
        sample = dataset[i]
        with open(save_path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {save_path}, {i+1}/{len(dataset)}")
