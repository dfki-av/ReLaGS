import os
import sys
import json
from configs.scan3r.rio import *
import configs.scan3r.define as scan3rdefine
from configs.scan3r.util_label import *
import trimesh
import numpy as np
import open3d as o3d



def search_scans_from_relationships(relationships, target_scan):
    for scan in relationships:
        if scan['scan'] == target_scan:
            return scan
    return None

def semseg_for_obj_ids(semseg, obj_ids):
    obj_id_to_semseg = {}
    for seg in semseg:
        for obj_id in obj_ids:
            if obj_id == seg['objectId']:
                obj_id_to_semseg[obj_id] = seg['label']
    return obj_id_to_semseg

def read_meta_files():
    with open(scan3rdefine.RELATION_CLS_FILE, 'r') as f:
        rel_classes = f.read().splitlines()
    #print(len(rel_classes))
    with open(scan3rdefine.CLASS160_FILE, 'r') as f:
        obj_classes = f.read().splitlines()

    with open(scan3rdefine.RELATION_FILE, 'r') as f:
        relationships = json.load(f)
    relationships = relationships['scans']

    rel_at_scene_dict = {}
    for rio, scan3r in rio_3dssg_mapping.items():
        rel_at_scene = search_scans_from_relationships(relationships, scan3r)
        rel_at_scene_dict[rio] = rel_at_scene
    return rel_classes, obj_classes, relationships, rel_at_scene_dict

def read_scene_gt(scene_pth):
    color_mesh_pth = os.path.join(scene_pth, scan3rdefine.OBJ_NAME)
    label_mesh_pth = os.path.join(scene_pth, scan3rdefine.LABEL_FILE_NAME)

    color_mesh = trimesh.load(color_mesh_pth, process=False)
    #label_mesh = trimesh.load(label_mesh_pth, process=False)
    
    points, obj_id_labels = read_pointcloud_R3SCAN(label_mesh_pth)
    obj_ids = np.unique(obj_id_labels)

    with open(os.path.join(scene_pth, scan3rdefine.SEMSEG_FILE_NAME), 'r') as f:
        semseg = json.load(f)
    semseg = semseg['segGroups']

    return points, obj_id_labels, semseg, obj_ids, color_mesh

def obj_ids_to_target_classes(obj_ids_to_raw_labels, label_name_mapping):
    obj_id_to_target_cls = {}
    for obj_id, raw_label in obj_ids_to_raw_labels.items():
        if raw_label in label_name_mapping:
            target_cls = label_name_mapping[raw_label]
        else:
            target_cls = 'other structure'
        obj_id_to_target_cls[obj_id] = target_cls
    return obj_id_to_target_cls

def visualize_relationship(points, obj_id_labels, obj_id_to_target_cls, sub_id, obj_id, rel_name):
    """
    Visualize the subject and object of a relationship with open3d pointcloud.
    """
    sub_points = points[(obj_id_labels == sub_id).reshape(-1)]
    obj_points = points[(obj_id_labels == obj_id).reshape(-1)]
    #print(f"Subject {obj_id_to_target_cls[sub_id]} points: {sub_points.shape[0]}")
    #print(f"Object {obj_id_to_target_cls[obj_id]} points: {obj_points.shape[0]}")
    sub_pcd = o3d.geometry.PointCloud()
    sub_pcd.points = o3d.utility.Vector3dVector(sub_points)
    sub_pcd.paint_uniform_color([1, 0, 0]) # Red
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 1, 0]) # Green
    o3d.visualization.draw_geometries(
        [sub_pcd, obj_pcd],
        window_name=f"{obj_id_to_target_cls[sub_id]} {rel_name} {obj_id_to_target_cls[obj_id]}"
    )

def read_splits():
    with open(scan3rdefine.TRAIN_SCANS_FILE, 'r') as f:
        train_scans = f.read().splitlines()
    with open(scan3rdefine.VAL_SCANS_FILE, 'r') as f:
        val_scans = f.read().splitlines()
    with open(scan3rdefine.TEST_SCANS_FILE, 'r') as f:
        test_scans = f.read().splitlines()
    return train_scans, val_scans, test_scans

def read_label_ply(scene_pth):
    label_mesh_pth = os.path.join(scene_pth, scan3rdefine.OBJ_NAME)
    label_mesh = trimesh.load(label_mesh_pth, process=False)
    return label_mesh

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()

    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]

    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])
        m_Height = int(data[3].strip().split(' ')[-1])
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(float(data[4].strip().split(' ')[-1]))
        m_Height = int(float(data[5].strip().split(' ')[-1]))
        m_Shift = int(float(data[6].strip().split(' ')[-1]))
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))

    m_frames_size = int(float(data[11].strip().split(' ')[-1]))

    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=np.matrix(m_intrinsic),
        m_frames_size=m_frames_size
    )

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split(' ')])
    return pose

def read_rgb_frame(frame_path):
    
    return rgb_image

def read_json(root, split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    selected_scans = selected_scans.union(read_txt_to_list(os.path.join(root, f'{split}_scans.txt')))
    with open(os.path.join(root, f"relationships_{split}.json"), "r") as read_file:
        data = json.load(read_file)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

def read_txt_to_list(file):
    output = []
    with open(file, 'r') as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output