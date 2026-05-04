ITERATION=0
CONTRIB=0.0005
TAU=3
DATA_DIR=/home/arafa/data/lerf_ovs
SAVE_FOLDER=lerf

for scan in figurines ramen teatime waldo_kitchen
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
        --iteration $ITERATION

done



