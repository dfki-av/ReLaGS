import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from configs.scan3r.rio import *
import configs.scan3r.define as scan3rdefine
from configs.scan3r.util_label import *
from configs.scan3r.util_metafile import *
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
import pickle
from pycocotools import mask as mask_utils
from tqdm.contrib.concurrent import process_map
from functools import partial

def get_object_frame(scan_name, scene_data, output_dir, vis=False):
    scan = scene_data[scan_name]
    obj_data = scan['obj']

    extrinsic_list, intrinsic_info, image_names = read_scan_info_R3SCAN(scan_name, mode='rgb')
    #print(image_names)
    image_width = intrinsic_info['m_Width']
    image_height = intrinsic_info['m_Height']
    scan_pth = os.path.join(define.SCAN3R_ROOT_PATH, scan_name, 'labels.instances.annotated.v2.ply')
    points, labels = read_pointcloud_R3SCAN(scan_pth)

    mesh_path = os.path.join(define.SCAN3R_ROOT_PATH, scan_name, scan3rdefine.LABEL_FILE_NAME)
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # --- Initialize visualizer ONCE globally ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540, visible=False)
    ctr = vis.get_view_control()
    params = o3d.camera.PinholeCameraParameters()

    # Replace the old mesh (remove and re-add)
    vis.clear_geometries()
    vis.add_geometry(mesh)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsic_info['m_Width'], intrinsic_info['m_Height'],
        intrinsic_info['m_intrinsic'][0,0], intrinsic_info['m_intrinsic'][1,1],
        intrinsic_info['m_intrinsic'][0,2], intrinsic_info['m_intrinsic'][1,2]
    )
    params.intrinsic = intrinsic

    # --- Helper to render one mesh sequence ---
    def render_mesh_depth_sequence(mesh_obj):
        vis.clear_geometries()
        vis.add_geometry(mesh_obj)
        depth_list, mask_list = [], []
        for i, extrinsic in enumerate(extrinsic_list):
            world_to_camera = np.linalg.inv(extrinsic)
            params.extrinsic = world_to_camera
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

            vis.update_geometry(mesh_obj)
            vis.poll_events()
            vis.update_renderer()

            # --- Capture depth only; color is loaded later by the packer if needed. ---
            depth = vis.capture_depth_float_buffer(do_render=True)
            depth_np = np.asarray(depth)

            mask_np = (depth_np > 0).astype(np.uint8)
            depth_list.append(depth_np)
            mask_list.append(mask_np)
        return depth_list, mask_list

    try:
        # --- 1. Render full mesh ---
        depth_full_mesh, _ = render_mesh_depth_sequence(mesh)

        # --- 2. Render all submeshes ---
        object2frame = dict()

        faces = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        flat_labels = labels.squeeze()
        frame_area = image_width * image_height
        large_structure_labels = {'floor', 'ceiling', 'wall'}

        for inst in obj_data.keys():
            object2frame[inst] = []

            inst_mask = (flat_labels == int(inst))
            face_mask = inst_mask[faces].all(axis=1)
            selected_faces = faces[face_mask]
            if len(selected_faces) == 0:
                continue

            unique_vertices, inverse_idx = np.unique(selected_faces.flatten(), return_inverse=True)
            selected_vertices = vertices[unique_vertices]
            remapped_faces = inverse_idx.reshape((-1, 3))

            submesh = o3d.geometry.TriangleMesh()
            submesh.vertices = o3d.utility.Vector3dVector(selected_vertices)
            submesh.triangles = o3d.utility.Vector3iVector(remapped_faces)

            if vertex_colors is not None:
                submesh.vertex_colors = o3d.utility.Vector3dVector(
                    vertex_colors[unique_vertices]
                )

            #print(f"Rendering submesh {inst} with {len(selected_vertices)} verts")
            depth_obj_mesh, mask_obj_mesh = render_mesh_depth_sequence(submesh)
            for i in range(len(depth_obj_mesh)):
                mask = mask_obj_mesh[i]
                obj_depth = depth_obj_mesh[i]
                full_depth_masked = depth_full_mesh[i] * mask
                depth_diff = np.abs(full_depth_masked - obj_depth)
                non_occluded_mask = (depth_diff < 0.005) & (mask == 1)  # 2cm tolerance

                pixel_num = int(non_occluded_mask.sum())
                is_large_structure = obj_data[inst] in large_structure_labels
                keep_frame = pixel_num > (20000 if is_large_structure else 500)

                # less than 1% of the frame area 960*540*0.05*0.05=1296 or large objects like floor, ceiling, wall less than 4% of the frame area
                if keep_frame:
                    ys, xs = np.nonzero(non_occluded_mask)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    ratio = round(float(pixel_num) / frame_area, 4)
                    visible_mask = non_occluded_mask.astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(visible_mask))
                    object2frame[inst].append((image_names[i],
                                               pixel_num,
                                               ratio,
                                               bbox,
                                               rle))
    finally:
        vis.destroy_window()

    # Save to pickle
    output_filepath = os.path.join(output_dir, f"{scan_name}_object2image.pkl")
    with open(output_filepath, "wb") as f:
        pickle.dump(object2frame, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_scan_info_R3SCAN(scan_id, mode='rgb'):
    scan_path = os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode=mode)

    extrinsic_list, frame_paths = [], []

    for i in range(0, intrinsic_info['m_frames_size'], 10):
        frame_paths.append("frame-%s." % str(i).zfill(6) + 'color.jpg')
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6) + "pose.txt")

        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic)

    return np.array(extrinsic_list), intrinsic_info, frame_paths


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3RScan preprocessing parameters")
    parser.add_argument("--root", default=None, type=str,
                        help="path of 3DSSG dataset")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="path to save the output results")
    parser.add_argument("--split", default="train", type=str,
                        help="dataset split to process (train/validation/test)")
    parser.add_argument("--workers", default=4, type=int,
                        help="number of worker processes")
    args = parser.parse_args(sys.argv[1:])
    
    if args.root is not None:
        root = args.root
    else:
        root = os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, "3DSSG_subset")
    
    scene_data, selected_scans = read_json(root, args.split)

    out_dir = args.output_dir if args.output_dir is not None else os.path.join(scan3rdefine.SCAN3R_ROOT_PATH, "scan3r_obj2frame")
    os.makedirs(out_dir, exist_ok=True)

    scans = []
    for scan in sorted(selected_scans):
        if scan not in scene_data:
            continue
        if os.path.exists(os.path.join(out_dir, f"{scan}_object2image.pkl")):
            print(f"Pickle for {scan} already exists, skipping...")
            continue
        scans.append(scan)
    
    process_map(partial(get_object_frame, scene_data=scene_data, output_dir=out_dir), scans, max_workers=args.workers, chunksize=4)

    exit()