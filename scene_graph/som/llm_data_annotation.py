import torch
import argparse
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from transformers import AutoModel
import json
import os
import glob
import os.path as osp
import pickle
from tqdm import tqdm
from openai import OpenAI
from scene_graph.som.visualizer import Visualizer

# set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True


# sam parameters
label_mode = '1'
text_size, hole_scale, island_scale=640,100,100
text, text_part, text_thresh = '','','0.0'
alpha = 0.15
anno_mode = ['Mask', 'Mark']


OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

gpt_client = OpenAI(api_key=OPEN_API_KEY)
gpt_model_name = "gpt-4o"


def load_image(image_path, rotate=0):
    image = Image.open(image_path).convert('RGB')
    if rotate != 0:
        image = image.rotate(rotate*90, expand=True)
    return image

def load_sam_masks(sam_mask_path, rotate=0):
    mask = np.load(sam_mask_path, allow_pickle=True)  # mask shape: (4, H, W)
    # Rotate spatial dimensions of the mask the same way as the image (90° clockwise)
    if rotate != 0:
        mask = np.rot90(mask, k=rotate, axes=(1, 2)).copy()
    return mask  # (4, H, W)

def rotate_mask_outputs(output, masks, rotate):
    
    output = np.rot90(output, k=rotate, axes=(0, 1)).copy()
    masks = np.rot90(masks, k=rotate, axes=(1, 2)).copy()

    return output, masks

def get_annotated_image(img, sam_mask, level, label_mode='1', alpha=0.1, anno_mode=['Mask']):
    unique_ids = np.unique(sam_mask[level][sam_mask[level] != -1])
    H, W = sam_mask.shape[1], sam_mask.shape[2]
    min_area = H*0.01 * W*0.01  # minimum area threshold 
    sam_mask_som_id_map = {}
    label = 1
    for uid in unique_ids:
        mask = sam_mask[3] == uid
        if mask.sum()>min_area:
            sam_mask_som_id_map[label] = int(uid)
            label += 1
    
    visual = Visualizer(np.asarray(img))
    for label, uid in sam_mask_som_id_map.items():
        mask = sam_mask[3] == uid
        demo = visual.draw_binary_mask_with_number(mask, text=str(label), alpha=alpha,
                                                   label_mode=label_mode, anno_mode=anno_mode)
    
    return demo.get_image(), sam_mask_som_id_map

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

def get_objects_dict(output_gpt, sam_mask_som_id_map):
    tag2class = output_gpt['objects']
    filtered_map = {}
    for tag, class_name in tag2class.items():
        tag_int = int(tag)
        if tag_int in sam_mask_som_id_map:
            filtered_map[tag] = sam_mask_som_id_map[tag_int]
    return tag2class, filtered_map

def get_relationships_dict(output_gpt):
    relation_dict = output_gpt['relationships_affordances']
    return relation_dict

def generate_gpt_dataset(data_dir, output_dir, prompt, redo=None, edit=False, rotate=0, level=3):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_names = os.listdir(os.path.join(data_dir, "images"))
    img_names = sorted(img_names)

    need_to_redo = []
    for img_idx in tqdm(range(0, len(img_names))):
        image_name = img_names[img_idx]
        image_path = os.path.join(data_dir, "images", image_name)
        sam_mask_path = os.path.join(data_dir, "language_features", image_name.replace(".jpg", "_s.npy"))
        image_base_name = os.path.splitext(image_name)[0]
        if redo and image_name not in redo:
            continue
        if os.path.exists(os.path.join(output_dir, f"{image_base_name}_som.png")):
            continue
        if not os.path.exists(sam_mask_path):
            print(f"Sam mask not found for {image_name}, skipping...")
            #need_to_redo.append(image_path)
            continue
        image = load_image(image_path, rotate=rotate)
        masks = load_sam_masks(sam_mask_path, rotate=rotate)
        
        som_image, sam_mask_som_id_map = get_annotated_image(image, masks, level=level, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        
        try:
            output_gpt = gpt_som_reasoning(som_image, prompt)
        except Exception:
            print('GPT reasoning failed!')
            output_gpt = '{"objects": {}, "relationship": []}'
        
        Image.fromarray(som_image).save(os.path.join(output_dir, f"{image_base_name}_som.png"))

        if rotate != 0:
            som_image, masks = rotate_mask_outputs(som_image, masks, -rotate)
        
        
        if redo and edit:
            with open(os.path.join(output_dir, f"{image_base_name}_gpt_output.txt"), 'r') as f:
                output_gpt = f.read()
        with open(os.path.join(output_dir, f"{image_base_name}_gpt_output.txt"), 'w') as f:
            f.write(output_gpt)
        try:
            structured_gpt_output = json.loads(output_gpt)
            tag2class, sam_mask_som_id_map = get_objects_dict(structured_gpt_output, sam_mask_som_id_map)
            relation_dict = get_relationships_dict(structured_gpt_output)
            
            anno_output = {}
            anno_output["objects"] = tag2class
            anno_output["relationships"] = relation_dict
            anno_output["object_map"] = sam_mask_som_id_map

            json.dump(anno_output, open(os.path.join(output_dir, f"{image_base_name}.json"), 'w'), indent=4)
            #json.dump(tag2class, open(os.path.join(output_dir, f"{image_base_name}_tag2class.json"), 'w'))
            #json.dump(relation_dict, open(os.path.join(output_dir, f"{image_base_name}_relation_dict.json"), 'w'))
        except Exception:
            print(f"Error in {image_path}")
            need_to_redo.append(image_path)
    print(f"Need to redo: {need_to_redo}")
    # save need_to_redo list
    with open(os.path.join(output_dir, "need_to_redo.txt"), 'w') as f:
        for item in need_to_redo:
            f.write("%s\n" % item)


def get_args():
    parser = argparse.ArgumentParser(description="Generate GPT dataset")
    parser.add_argument("--data_dir", type=str, help="path to nerfstudio dataset")
    parser.add_argument("--redo", type=str, default=None, help="comma separated list of image names to redo")
    parser.add_argument("--edit", action="store_true", help="Edit the gpt output")
    parser.add_argument("--out_dir", type=str, default="chatgpt")
    parser.add_argument("--rel_types", type=str, default="semantic", help="either [semantic] or [affordance]")
    parser.add_argument("--rotate", type=int, default=0, help="rotate image clockwise by 90 degrees x times")
    parser.add_argument("--mode", type=str, default="gpt", help="either [gpt] or [gpt_redo]")
    parser.add_argument("--level", type=int, default=3, help="SAM mask level to use")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    
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
        generate_gpt_dataset(data_dir, chatgpt_output_dir, chat_gpt_prompt, redo=args.redo, edit=args.edit, rotate=args.rotate, level=args.level)
    