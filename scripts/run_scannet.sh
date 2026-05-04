ITERATION=0
CONTRIB=0.0005
TAU=3
DATA_DIR=/home/arafa/data/scanNet
SAVE_FOLDER=scannet

for scan in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00
do

python max_weight_pruning.py \
    -s ${DATA_DIR}/${scan}/ \
    -m ./output/lerf/${scan} \
    --contribution_threshold $CONTRIB



python sp_partition.py \
            -s ${DATA_DIR}/${scan}/ \
            -m ./output/lerf/${scan} \
            --iteration $ITERATION \
            -a

python graph_weight.py \
    --iteration $ITERATION \
    -s ${DATA_DIR}/${scan}/ \
    -m ./output/lerf/${scan} \
    --config ./configs/lerf.yml \
    --level 1

python sp_partition.py \
    -s ${DATA_DIR}/${scan}/ \
    -m ./output/lerf/${scan} \
    --iteration $ITERATION \
    -k neighbor_new.pt \
    --pcp_regularization 0.1 \
    --pcp_spatial_weight 0.1 \
    -a

python merge_proj.py \
        -s ${DATA_DIR}/${scan}/ \
        -m ./output/lerf/${scan} \
        --thres_connect 0.9,0.7,0.7 \
        --thres_merge 20 \
        --tau $TAU \
        --vlm_type openseg \
        --iteration $ITERATION

python predict_3d_scene_graph.py \
        --root_pred ./output/scannet/${scan}/ \
        --iteration $ITERATION --neighbor_thresh 5.0 \
        --vlm_type openseg

done



