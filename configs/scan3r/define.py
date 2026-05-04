# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os


RIO_ROOT_PATH = "/home/xie/Documents/datasets/3DSSG_processed/"
SCAN3R_ROOT_PATH = "/netscratch/arafa/data/3DSSG_processed/"


Scan3RJson_PATH = RIO_ROOT_PATH+'3RScan.json'
LABEL_MAPPING_FILE = RIO_ROOT_PATH+'3RScan.v2 Semantic Classes - Mapping.csv'
CLASS160_FILE = RIO_ROOT_PATH+'classes160.txt'
RELATION_CLS_FILE = RIO_ROOT_PATH+'relationships.txt'
RELATION_FILE = RIO_ROOT_PATH+'relationships.json'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.v2.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

# 3RScan sequence file names
FRAME_EXT = '.color.jpg'
DEPTH_EXT = '.depth.pgm'
POSE_EXT = '.pose.txt'
CAMERA_FILE_NAME = '_info.txt'

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'


NAME_SAME_PART = 'same part'

# Splits
TRAIN_SCANS_FILE = SCAN3R_ROOT_PATH + 'train_scans.txt'
VAL_SCANS_FILE = SCAN3R_ROOT_PATH + 'val_scans.txt'
TEST_SCANS_FILE = SCAN3R_ROOT_PATH + 'test.txt'