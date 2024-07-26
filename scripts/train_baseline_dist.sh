#!/bin/sh
# ./scripts/train_baseline_dist.sh graphvlad triplet vgg16 pitts 30k

PYTHON=${PYTHON:-"python"}
GPUS=1
DATE=$(date '+%d-%b') 
METHOD="$1"
LOSS="$2"
ARCH="$3"
DATASET="$4"
SCALE="$5"
DATE=$(date '+%d-%b')
NUMCLUSTER=64

LAYERS=conv5
LR=0.001


# FILES="/home/leo/usman_ws/models/openibl/fastscnn/${ARCH}-${METHOD}-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"
INIT_DIR="/home/leo/usman_ws/datasets/openibl-init"
FAST_SCNN="/home/leo/usman_ws/datasets/fast_scnn/fast_scnn_citys.pth"
DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"
FILES="/home/leo/usman_ws/models/openibl/vgg16-graphvlad-triplet-pitts30k-lr0.001-tuple1-11-Jun/"
# FILES="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.0001-tuple1-12-Apr/"

echo ${FILES}


if [ $# -ne 5 ]
  then    
    echo "./scripts/train_baseline_dist.sh graphvlad triplet vgg16 pitts 30k"
    exit 1
fi
PORT=6010
# echo "==========Starting Training============="
# echo "========================================"
# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
#   -d ${DATASET} --scale ${SCALE} \
#   -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
#   --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
#   --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
#   --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
#   --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
#   --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
#    --method ${METHOD}

echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test_pitts_tokyo.py --launcher pytorch \
    -a ${ARCH} --test-batch-size 32 -j 4 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME} --fast-scnn ${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}
done
