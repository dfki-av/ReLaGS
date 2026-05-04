import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from random import randint
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import numpy as np
from utils.vlm_utils import ClipSimMeasure, DINOV3SimMeasure
from gaussian_renderer import render_2d_with_ids, render
from collections import defaultdict
import glob
from collections import Counter
from utils.scenegraph_utils import obb_min_distance, get_obb
from transformers import AutoModel

obb_thres = 5.0
center_thres = 10.0

def get_jina_feature(relationships: Counter, jina_encoder):
    text_list = list(relationships)
    text_features = torch.from_numpy(jina_encoder(text_list)).cuda()
    text_features = torch.nn.functional.normalize(text_features, dim=1, p=2)
    
    # weighted sum with softmaxed counter nums 
    # e.g. counter {"on": 4, "next to": 1, "beside": 1} -> weights [0.9094, 0.0453, 0.0453]
    weights = torch.softmax(torch.tensor([relationships[t] for t in text_list], dtype=torch.float32, device="cuda"), dim=0)
    weighted_feature = (text_features * weights.unsqueeze(1)).sum(0)
    weighted_feature = weighted_feature
    return weighted_feature

def load_som_annos(sg_anno_path):
    sg_annos = {}
    # list all json files in the folder and sort them
    json_files = [f for f in os.listdir(sg_anno_path) if f.endswith('.json')]
    json_files.sort()

    for file in json_files:
        frame_name = file.replace(".json", "")
        with open(os.path.join(sg_anno_path, file), 'r') as f:
            sg_annos[frame_name] = json.load(f)
    print("loaded scene graph annos for {} frames".format(len(sg_annos)))
    return sg_annos

def occlusion_aware_mask_rendering(gaussians, mask, cam, pipe, background):
    point_valid = mask.expand(-1, 20).cuda()
    gaussians._semantics = point_valid        
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]

    w, h = cam.image_width, cam.image_height
    mask = embd_sim.reshape(20, -1)[0] > 0.5
    binary_mask = mask.reshape(h, w)
    return binary_mask

@torch.no_grad()
def create_and_merge_from_2D(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, load_sem=True, shuffle=False, load_test=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene_graph_path = args.output_folder

    scene_name = os.path.basename(dataset.model_path)
    os.makedirs(os.path.join(scene_graph_path, scene_name), exist_ok=True)
    print("scene name:", scene_name)

    if args.vlm_type == 'clip':
            nag = torch.load(os.path.join(dataset.model_path,  f"sai_nag.pt"))
            vlm = ClipSimMeasure()
            vlm.load_model()
    elif args.vlm_type == 'dinov3':
        nag = torch.load(os.path.join(dataset.model_path,  f"sai_nag_dinov3.pt"))
        vlm = DINOV3SimMeasure()
        vlm.load_model()
    elif args.vlm_type == 'openseg':
        nag = torch.load(os.path.join(dataset.model_path,  f"sai_nag_openseg.pt"))
        vlm = ClipSimMeasure(clip_type="ViT-L-14-336")
        vlm.load_model()
    else:
        raise NotImplementedError(f"Unknown vlm type {args.vlm_type}")

    nag_indices = nag['nag'][-1].cuda() # l3

    if len(nag['nag'][-1].shape) != gaussians._xyz.shape[0:1]:
        assert "nag indices shape mismatch"
    
    
    superpoint_indices_at3 = torch.unique(nag['nag'][-1].cuda())

    if not args.skip_lift:
        sg_anno_path = os.path.join(dataset.model_path, "chatgpt")
        sg_annos = load_som_annos(sg_anno_path)

        # Looping through each view and get 2D anno - 3D scene association of each view 
        view_stacks = scene.getTrainCameras().copy()
        for i, cam in enumerate(view_stacks):
            if cam.image_name not in sg_annos:
                print("skip view:", cam.image_name)
                continue

            # get scene graph anno for this view
            sg_objects = sg_annos[cam.image_name]["objects"]
            sg_relationships = sg_annos[cam.image_name]["relationships"]
            sg_anno_2_sam3_map = sg_annos[cam.image_name]["object_map"]
            if len(sg_objects) == 1 or len(sg_relationships) == 0 or len(sg_anno_2_sam3_map) != len(sg_objects):
                print("skip view due to insufficient objects or relationships")
                continue
            
            # get sam mask ground truth at level 3 for this view
            feature_level = -1 # l3
            sam_mask_l3 = cam.semantic["seg_map"][feature_level]
            render_pkg = render_2d_with_ids(cam, gaussians, pipe, background, args)
            impact_id = render_pkg["impact_id"].to(torch.int64)
            
            # 1. First create annotation object id to superpoint id map
            obj_sp_id_map = {} # map from object id in sam mask at level 3 to superpoint id in 3D
            for i, obj_id in enumerate(sg_objects):
                sam_mask_id = sg_anno_2_sam3_map[obj_id]
                mask = (sam_mask_l3 == sam_mask_id)
                inmask_impact_gs_ids = impact_id[:,mask]
                unique_inmask_impact_gs_ids = torch.unique(inmask_impact_gs_ids)
                unique_inmask_impact_gs_ids = unique_inmask_impact_gs_ids[unique_inmask_impact_gs_ids > 0]
                obj_sp_id_map[obj_id] = {"level_3": None, "level_2": None}
                
                # 1.1 Get level 3 superpoint id
                level = 3
                nag_indices_in_view = nag['nag'][level][unique_inmask_impact_gs_ids]
                nag_indices_in_view_unique, counts = torch.unique(nag_indices_in_view, return_counts=True)
                values, pos = torch.topk(counts, k=min(3, counts.shape[0]))
                sp_id = nag_indices_in_view_unique[pos][0].item()
                obj_sp_id_map[obj_id]["level_3"] = sp_id

                # 1.2 Get level 2 superpoint id
                level = 2
                nag_indices_in_view = nag['nag'][level][unique_inmask_impact_gs_ids]
                nag_indices_in_view_unique, counts = torch.unique(nag_indices_in_view, return_counts=True)
                values, pos = torch.topk(counts, k=min(3, counts.shape[0]))
                sp_id = nag_indices_in_view_unique[pos][0].item()
                obj_sp_id_map[obj_id]["level_2"] = sp_id
            
            # 2. Then create edges based on relationships
            edge_dict = {"level_3": defaultdict(list), "level_2": defaultdict(list)}
            for rel in sg_relationships:
                subject_id = str(rel["s_id"])
                object_id = str(rel["o_id"])
                predicate = rel["predicate"] if "predicate" in rel else rel["predicates"]
                predicate = predicate.replace("_", " ").replace("/", " ")
                predicates = [ele.lstrip(' ') for ele in predicate.split(",")]
                sp_s = obj_sp_id_map[subject_id]["level_3"]
                sp_o = obj_sp_id_map[object_id]["level_3"]
                if sp_s == sp_o:
                    #print("skip self loop edge at level 3:", sp_s, sp_o)
                    sp_s = obj_sp_id_map[subject_id]["level_2"]
                    sp_o = obj_sp_id_map[object_id]["level_2"]
                    if sp_s != sp_o:
                        edge_dict["level_2"][(sp_s, sp_o)].extend(predicates)
                else:
                    edge_dict["level_3"][(sp_s, sp_o)].extend(predicates)
            
            print("final edges:", edge_dict)
            if len(edge_dict["level_3"]) == 0 and len(edge_dict["level_2"]) == 0:
                print("skip view due to no valid edges")
            else:
                torch.save(edge_dict, os.path.join(scene_graph_path, scene_name, cam.image_name, f"edges.pt"))

    if not args.skip_merge:
        scene_name = os.path.basename(dataset.model_path)
        single_view_sg_root = os.path.join(scene_graph_path, scene_name)
        single_view_sg_paths = glob.glob(os.path.join(single_view_sg_root, "*", "edges.pt"))
        single_view_sg_paths = sorted(single_view_sg_paths)

        combined_edge_dict = {"level_3": defaultdict(list), "level_2": defaultdict(list)}
        for sg_path in single_view_sg_paths:
            edge_dict = torch.load(sg_path, weights_only=False)
            for level in edge_dict.keys():
                edge_dict_at_level = edge_dict[level]
                for key in edge_dict_at_level:
                    a, b = key
                    if a == b:
                        print("skip self loop edge:", key)
                        continue
                    combined_edge_dict[level][key].extend(edge_dict[level][key])
        
        for level in combined_edge_dict.keys():
            combined_edge_dict_at_level = combined_edge_dict[level]
            for key in combined_edge_dict_at_level:
                combined_edge_dict[level][key] = Counter(combined_edge_dict[level][key])

        # save combined edges
        os.makedirs(os.path.join(scene_graph_path, scene_name), exist_ok=True)
        torch.save(combined_edge_dict, os.path.join(dataset.model_path, "combined_edges_ml.pt"))

        if not args.skip_jina:
            print("Loading Jina model...")
            jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(torch.bfloat16).cuda()
            jina_encode =  lambda x: jina_model.encode(x, task='text-matching', truncate_dim=512)
            print("Loaded Jina model.")

            # Get obb and center of each superpoint
            obj_center_dict = {}
            for sp_idx in superpoint_indices_at3:

                selected_mask = (nag_indices == sp_idx)
                sp_pts = gaussians._xyz[selected_mask].detach().cpu().numpy()
                center = sp_pts.mean(0)
                obb = get_obb(sp_pts, remove_outlier=True)
                obj_center_dict[sp_idx.item()] = {"center": center, "obb": obb}
            
            filtered_jina_edges = {}
            edge_nums = 0

            for e in combined_edge_dict.keys():
                id_a, id_b = e
                line_pts = [obj_center_dict[id_a]["center"], obj_center_dict[id_b]["center"]]
                line_length = np.linalg.norm(line_pts[0] - line_pts[1])
                bbox_distance = obb_min_distance(obj_center_dict[id_a]["obb"], obj_center_dict[id_b]["obb"], early_exit_tol=0.0)[0]
                # Filter out edges that are too far away
                if bbox_distance > obb_thres or line_length > center_thres:
                    continue
                filtered_jina_edges[e] = get_jina_feature(combined_edge_dict[e], jina_encode)
                edge_nums += 1

            # Save filtered edges with jina features
            print("Total edges:", edge_nums, "Filtered edges:", len(combined_edge_dict)-edge_nums)
            torch.save(filtered_jina_edges, os.path.join(dataset.model_path, f"jina_edges_ml.pt"))
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_lift", action='store_true', default=False, help="whether to skip the lifting step")
    parser.add_argument("--skip_merge", action='store_true', default=False, help="whether to skip the merging step")
    parser.add_argument("--skip_jina", action='store_true', default=False, help="whether to skip converting text to jina features")
    parser.add_argument("--fill_knn", action='store_true', default=False, help="whether to use knn to fill local edges")
    parser.add_argument("--output_folder", default="./scene_graph_temp_imgs")
    parser.add_argument("--anno_folder", default=None)
    parser.add_argument("--vlm_type", default="openseg")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    safe_state(True)

    create_and_merge_from_2D(lp.extract(args), pp.extract(args), args)
