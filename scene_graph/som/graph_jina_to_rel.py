import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import numpy as np
from transformers import AutoModel
from collections import Counter
import open_clip

R3SCAN_RAW = _REPO_ROOT / "configs" / "3RScan_meta"
class_names = np.array([l.rstrip() for l in open(os.path.join(R3SCAN_RAW, "class.txt"), 'r')])
relationships_names = np.array([l.rstrip() for l in open(os.path.join(R3SCAN_RAW, "relationships.txt"), 'r')])

# Load Jina and CLIP models
jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(torch.bfloat16).cuda()
jina_encode =  lambda x: jina_model.encode(x, task='text-matching', truncate_dim=512)
clip_model, _, _ = open_clip.create_model_and_transforms(
                                        "ViT-B-16", 
                                        pretrained="laion2b_s34b_b88k")
clip_model = clip_model.eval().to("cuda")
clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")


def jina_to_text(args):
    scene_name = args.scene_name
    text_edges = torch.load(os.path.join("./scene_graph_temp_imgs",scene_name, f"combined_edges.pt"), weights_only=False)
    jina_edges = torch.load(os.path.join("./scene_graph_temp_imgs",scene_name, f"jina_edges.pt"), weights_only=False)

    gt_text_features = torch.from_numpy(jina_encode(relationships_names)).cuda()

    for e in jina_edges.keys():
        id_a, id_b = e
        edge_text = text_edges.get(e, None)
        if len(edge_text) <= 1:
            continue
        jina_feature = jina_edges[e]
        text_sim = torch.matmul(gt_text_features, jina_feature.unsqueeze(1))
        top3_idx = torch.topk(text_sim.squeeze(1), k=3).indices.cpu().numpy()
        print("Edge:", e, edge_text, "Learned", relationships_names[top3_idx], text_sim[top3_idx].squeeze().cpu().numpy())

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scene_name", type=str, default="ramen", help="Name of the scene to process.")
    parser.add_argument("--scene_root", type=str, default='./scene_graph_temp_imgs', help="Path to the scenes.")
    
    args = parser.parse_args()
    jina_to_text(args)
