#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import torch
import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic: torch.Tensor = None
    semantic_path: str = None
    sharpness: float = 0.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, load_img=False, load_sem=True, vlm_type='clip'):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) if load_img else None

        dinov3_path = os.path.join(images_folder, f'../language_features_dinov3/{image_name}_f_dinov3.npy')
        openseg_path = os.path.join(images_folder, f'../language_features_openseg/{image_name}_f_openseg.npy')
        semantic = None
        #vlm = 'clip'
        if load_sem:
            if vlm_type == 'clip':
                seg_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_s.npy'))).long()
                feature_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_f.npy')))
                semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
            elif vlm_type == 'dinov3':
                seg_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_s.npy'))).long()
                feature_map = torch.from_numpy(np.load(dinov3_path))
                semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
            elif vlm_type == 'openseg':
                seg_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_s.npy'))).long()
                feature_map = torch.from_numpy(np.load(openseg_path).astype(np.float32))
                semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
            else:
                raise NotImplementedError(f"VLM {vlm_type} not implemented")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              semantic=semantic, semantic_path=dinov3_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, load_img=False, load_sem=True, vlm_type='clip'):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), load_img=load_img, load_sem=load_sem, vlm_type=vlm_type)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", load_img=False, load_sem=True, vlm_type='clip'):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fov_y_exists = False
        if "camera_angle_x" in contents:
            fovx = contents["camera_angle_x"]
        else:
            focal_length_x = contents["fl_x"]
            focal_length_y = contents["fl_y"]
            
            fovx = focal2fov(focal_length_x, contents["w"])
            fovy = focal2fov(focal_length_y, contents["h"])
            
            fov_y_exists = True
        
        Width = contents["w"]
        Height = contents["h"]
        
        frames = contents["frames"]
        for idx, frame in tqdm.tqdm(enumerate(frames), total=len(frames), desc="Reading cameras from transforms"):
            if frame["file_path"].split(".")[-1].lower() in ["png", "jpg", "jpeg"]:
                cam_name = os.path.join(path, frame["file_path"])
            else:
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            if np.isnan(R).any() or np.isnan(T).any():
                continue
            
            image_name = Path(cam_name).stem
            image_path = os.path.join(path, cam_name)
            if load_img:
                image = Image.open(image_path)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                image = None
            
            if not fov_y_exists:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            dinov3_path = os.path.join(path, f'language_features_dinov3/{image_name}_f_dinov3.npy')
            openseg_path = os.path.join(path, f'language_features_openseg/{image_name}_f_openseg.npy')
            #vlm = 'clip'
            if load_sem:
                #sem_path = os.path.join(path, f'clip_feat/{idx + 1}.pt')
                if vlm_type == 'clip':
                    try:
                        seg_map = torch.from_numpy(np.load(os.path.join(path, f'language_features/{image_name}_s.npy'))).long()
                        feature_map = torch.from_numpy(np.load(os.path.join(path, f'language_features/{image_name}_f.npy')))
                        semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
                    except:
                        semantic = {'seg_map': None, 'fg_mask': None, 'sem': None}
                    
                elif vlm_type == 'dinov3':
                    try:
                        seg_map = torch.from_numpy(np.load(os.path.join(path, f'language_features/{image_name}_s.npy'))).long()
                        feature_map = torch.from_numpy(np.load(dinov3_path))
                        semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
                    except:
                        semantic = {'seg_map': None, 'fg_mask': None, 'sem': None}
                elif vlm_type == 'openseg':
                    try:
                        seg_map = torch.from_numpy(np.load(os.path.join(path, f'language_features/{image_name}_s.npy'))).long()
                        feature_map = torch.from_numpy(np.load(openseg_path)).float()
                        #print(feature_map.shape, feature_map.dtype, torch.isnan(feature_map).sum())
                        semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
                    except:
                        semantic = {'seg_map': None, 'fg_mask': None, 'sem': None}
                else:
                    raise NotImplementedError(f"VLM {vlm_type} not implemented")
            
            if "sharpness" in frame:
                sharpness = frame["sharpness"]
                #if sharpness < 15:
                    #print("Warning: image {} has low sharpness {}".format(image_name, sharpness))
                    #continue
            else:
                sharpness = 0.0
            if load_sem and semantic['seg_map'] is None:
                continue

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=Width, height=Height,
                            semantic=semantic, semantic_path=dinov3_path, sharpness=sharpness))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", load_img=True, load_sem=True, vlm_type='clip'):
    if os.path.exists(os.path.join(path, "transforms_train.json")):
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, vlm_type=vlm_type, load_img=load_img)
        print("Reading Test Transforms")
        # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, vlm_type=vlm_type, load_img=load_img)
        test_cam_infos = []
    # for 3RSCAN dataset
    else:
        print("Reading Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension, vlm_type=vlm_type, load_img=load_img)
        test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    txt_path = os.path.join(path, "points3d.txt")
    # for 3RSCAN dataset
    if os.path.exists(os.path.join(path, "labels.instances.align.annotated.v2.ply")):
        ply_path = os.path.join(path, "labels.instances.align.annotated.v2.ply")
    
    if not os.path.exists(ply_path) and os.path.exists(txt_path):
        print("Converting points3d.txt to .ply, will happen only the first time you open the scene.")
        xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}