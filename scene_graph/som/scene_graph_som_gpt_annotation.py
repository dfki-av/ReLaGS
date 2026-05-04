import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from gaussian_renderer import render, render_2d_with_ids
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
import numpy as np
from scene_graph.som.visualizer import Visualizer
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm
from openai import OpenAI

# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

label_mode = '1'
text_size, hole_scale, island_scale=640,100,100
text, text_part, text_thresh = '','','0.0'
alpha = 0.15
anno_mode = ['Mask', 'Mark']

OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

gpt_client = OpenAI(api_key=OPEN_API_KEY)
gpt_model_name = "gpt-4o"

def get_objects_dict(output_gpt):
    tag2class = output_gpt['objects']
    return tag2class

def get_relationships_dict(output_gpt):
    relation_dict = output_gpt['relationships_affordances']
    return relation_dict

def rotate_mask_outputs(output, masks, rotate):
    
    output = np.rot90(output, k=rotate, axes=(0, 1)).copy()
    masks = np.rot90(masks, k=rotate, axes=(1, 2)).copy()

    return output, masks

def gpt_som_reasoning(som_image, prompt):
    buffered = BytesIO()
    Image.fromarray(som_image).save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = gpt_client.chat.completions.create(
        model=gpt_model_name,
        response_format={ "type": "json_object" },
        temperature=0,
        seed=0,
        messages=[
            {
            "role": "system",
            "content": "You are an articulate assistant designed to output JSON, that describes the objects and their relationships in the image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                        "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]  
    )
    output_gpt = response.choices[0].message.content
    return output_gpt

@torch.no_grad()
def cluster_to_mask(dataset, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, load_sem=False, load_img=True, shuffle=False, load_test=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene_pred_pth = args.model_path
    if args.vlm_type == 'clip':
        nag_pth =os.path.join(scene_pred_pth,  f"sai_nag.pt")
    elif args.vlm_type == 'dinov3':
        nag_pth = os.path.join(scene_pred_pth,  f"sai_nag_dinov3.pt")

    elif args.vlm_type == 'openseg':
        nag_pth = os.path.join(scene_pred_pth,  f"sai_nag_openseg.pt")
    else:
        raise NotImplementedError(f"Unknown vlm type {args.vlm_type}")
    
    nag = torch.load(nag_pth)

    view_stack = scene.getTrainCameras()
    
    level = 3
    nag_indices = nag['nag'][level]
    cluster_ids = torch.unique(nag_indices)

    print("Number of clusters at level {}: {}".format(level, cluster_ids.shape[0]))
    som_images = {}
    for i, cam in enumerate(view_stack):
        original_image = cam.original_image.permute(1,2,0).cpu().numpy()*255
        height, width = original_image.shape[:2]
        sam_mask_som_id_map = {}
        render_pkg = render_2d_with_ids(cam, gaussians, pipe, background, args)
        impact_id = render_pkg["impact_id"].to(torch.int64)
        gaussian_ids = torch.unique(impact_id)
        
        visible_cluster_ids, counts = torch.unique(nag_indices[gaussian_ids], return_counts=True)
        min_per_frame_pixels = height * width * 0.05*0.05
        visible_cluster_ids = visible_cluster_ids.cpu().numpy().tolist()

        print("processing frame", cam.image_name, len(visible_cluster_ids), "visible clusters in view", i, "total clusters:", cluster_ids.shape[0], end="\r")
        
        for j in visible_cluster_ids:
            cluster_points = torch.zeros_like(nag_indices).float()
            cluster_points[nag_indices == j] = 1.0
            point_valid = cluster_points.unsqueeze(dim=1).expand(-1, 20).cuda()
            gaussians._semantics = point_valid        
            embd_sim = render(cam, gaussians, pipe, background)["semantics"]
            mask = embd_sim.reshape(20, -1)[0] > 0.5
            
            if mask.sum() < min_per_frame_pixels:
                continue
            sam_mask_som_id_map[str(int(j))] = mask.cpu().numpy().astype(np.uint8).reshape(height, width)

        visual = Visualizer(np.asarray(original_image))
        for label, mask in sam_mask_som_id_map.items():
            demo = visual.draw_binary_mask_with_number(mask, text=str(label), alpha=alpha,
                                                    label_mode=label_mode, anno_mode=anno_mode)
                
        som_image = demo.get_image()
        som_images[cam.image_name] = som_image
    
    return som_images



def generate_gpt_dataset(data_dir, output_dir, prompt, redo=None, edit=False, rotate=0, level=3, debug=False):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = os.listdir(os.path.join(data_dir, "images"))
    img_names = sorted(img_names)

    som_images = cluster_to_mask(lp.extract(args), pp.extract(args), args)

    if debug:
        stop_at = 5
    need_to_redo = []
    for img_idx in tqdm(range(0, len(img_names))):
        if debug and img_idx >= stop_at:
            break
        image_name = img_names[img_idx]
        image_path = os.path.join(data_dir, "images", image_name)
        
        image_base_name = os.path.splitext(image_name)[0]
        
        if redo and image_name not in redo:
            continue
        if os.path.exists(os.path.join(output_dir, f"{image_base_name}_som.png")):
            continue
        
        som_image = som_images[image_base_name]
        Image.fromarray(som_image).save(os.path.join(output_dir, f"{image_base_name}_som.png"))
        
        try:
            output_gpt = gpt_som_reasoning(som_image, prompt)
        except Exception:
            print('GPT reasoning failed!')
            output_gpt = '{"objects": {}, "relationship": []}'
        
        if rotate != 0:
            som_image, masks = rotate_mask_outputs(som_image, masks, -rotate)
        
        
        if redo and edit:
            with open(os.path.join(output_dir, f"{image_base_name}_gpt_output.txt"), 'r') as f:
                output_gpt = f.read()
        with open(os.path.join(output_dir, f"{image_base_name}_gpt_output.txt"), 'w') as f:
            f.write(output_gpt)
        try:
            structured_gpt_output = json.loads(output_gpt)
            tag2class = get_objects_dict(structured_gpt_output)
            relation_dict = get_relationships_dict(structured_gpt_output)
            
            anno_output = {}
            anno_output["objects"] = tag2class
            anno_output["relationships"] = relation_dict
            #anno_output["object_map"] = sam_mask_som_id_map
            json.dump(anno_output, open(os.path.join(output_dir, f"{image_base_name}.json"), 'w'), indent=4)

        except Exception:
            print(f"Error in {image_path}")
            need_to_redo.append(image_path)
    print(f"Need to redo: {need_to_redo}")
    # save need_to_redo list
    with open(os.path.join(output_dir, "need_to_redo.txt"), 'w') as f:
        for item in need_to_redo:
            f.write("%s\n" % item)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--vlm_type', type=str, default='clip', choices=['clip', 'dinov3', 'openseg'],
                        help='which vlm type to use')
    parser.add_argument("--data_dir", type=str, help="path to nerfstudio dataset")
    parser.add_argument("--redo", type=str, default=None, help="comma separated list of image names to redo")
    parser.add_argument("--edit", action="store_true", help="Edit the gpt output")
    parser.add_argument("--out_dir", type=str, default="chatgpt_")
    parser.add_argument("--rel_types", type=str, default="semantic", help="either [semantic] or [affordance]")
    parser.add_argument("--rotate", type=int, default=0, help="rotate image clockwise by 90 degrees x times")
    parser.add_argument("--mode", type=str, default="gpt", help="either [gpt] or [gpt_redo]")
    parser.add_argument("--level", type=int, default=3, help="SAM mask level to use")
    parser.add_argument("--debug", action="store_true", help="Run on a small subset of frames")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])


    if args.rel_types == "semantic":
        chat_gpt_prompt = """
            1. Object Identification: Identify all objects in the image by their tag. Create a dict that maps tag_id to class_name.

            2. Affordance/Relationship Detection: For every pair of tagged objects that are clearly related, describe the semantic relationships and affordances as a list of dictionaries using the format [s_id: #n1, subject_class: x, o_id: #n2, object_class: y, predicates: [p1, p2, ...]]. For subjects and objects sharing multiple relationships/affordances, concatenate predicates with a comma in the [predicate] field.

            - Avoid generic terms like "next to" for ambiguous relationships. Instead, specify relationships with precise relationships and affordances describing spatial relationships [over/under etc.], comparative relationships [larger/smaller than, similar/same type/color], functional relationships [part of/belonging to, turns on], support relationships [standing on, hanging on, lying on, attached to].
            - Do not use left/right, always use 3D consistant relationships.
            - Always combine a spatial relationship with a semantic, comparative, functional or support relationship using a comma (e.g., [A] [above, lying on] [B]).
            - For symmetrical relationships, include both directions (e.g., [A] [above] [B] and [B] [below] [A]).
            - Even for distant objects highlight if they are [same/similar/same color/same object type]
            Example Output:

            objects = [4: floor, 7: table, 12: chair, ...]

            relationships_affordances = [
                [s_id: 4, subject_class: table, o_id: 7, object_class: floor, predicates: standing on],
                [s_id: 12, subject_class: chair, o_id: 13, object_class: chair, predicates: next to, same as],
                [s_id: 6, subject_class: pillow, o_id: 8, object_class: couch, predicates: belongs to],
                [s_id: 7, subject_class: floor, o_id: 3, object_class: carpet, predicates: under],
                [s_id: 3, subject_class: carpet, o_id: 7, object_class: floor, predicates: above, lying on],
                [s_id: 9, subject_class: table, o_id: 14, object_class: table, predicates: bigger than],
                ...
            ]
        """
    elif args.rel_types == "affordance":
        chat_gpt_prompt = """
            1. Object Identification: Identify all objects in the image by their tag. Create a dict that maps tag_id to class_name.

            2. Inter-object Affordance/Action Detection: For every pair of tagged objects that are clearly have a shared affordance, describe the affordances/actions as a list of dictionaries using the format [s_id: #n1, subject_class: x, o_id: #n2, object_class: y, affordance: [a1, a2, ...]]. For subjects and objects sharing multiple affordances, concatenate affordances with a comma in the [affordance] field.
            - Only state what is observed in the scene, do not invent affordances.
            - For symmetrical affordances, include both directions (e.g., [A] [heats up] [B] and [B] [is being heated up] [A]).
            - Even for distant objects highlight if they have a general affordance like [belongs to] or [can be organized in].
            Example Output:

            objects = [4: lamp, 7: light switch, 12: remote, ...]

            relationships_affordances = [
                [s_id: 7, subject_class: light switch, o_id: 4, object_class: lamp, predicates: turns on],
                [s_id: 12, subject_class: remote, o_id: 13, object_class: TV, predicates: controls],
                [s_id: 6, subject_class: wall socked, o_id: 8, object_class: toaster, predicates: connectes to],
                [s_id: 9, subject_class: shoe, o_id: 14, object_class: shoe rack, predicates: belongs to],
                [s_id: 2, subject_class: stove, o_id: 3, object_class: kettle, predicates: heats up],
                [s_id: 8, subject_class: twol, o_id: 17, object_class: washing machine, predicates: gets cleaned by],
                ...
            ]
        """
    else:
        raise NotImplementedError("not implemented prompting strategy")


    if args.mode == "gpt" or args.mode == "gpt_redo":
        # in case there was a processing mistake and a manual correction
        data_dir = args.data_dir
        chatgpt_output_dir = os.path.join(data_dir, args.out_dir)
        if not os.path.exists(os.path.join(chatgpt_output_dir)):
            os.makedirs(os.path.join(chatgpt_output_dir))
        generate_gpt_dataset(data_dir, chatgpt_output_dir, chat_gpt_prompt, redo=args.redo, edit=args.edit, rotate=args.rotate, level=args.level, debug=args.debug)

