#!/bin/sh
# ./scripts/train_baseline_dist.sh graphvlad triplet vgg16 pitts 30k

PYTHON=${PYTHON:-"python"}
GPUS=1
DATE=$(date '+%d-%b') 
METHOD="$1"
ARCH="$2"
DATASET="$3"
SCALE="$4"
DATE=$(date '+%d-%b')
NUMCLUSTER=64

LAYERS=conv5
LR=0.001



INIT_DIR="/home/leo/usman_ws/datasets/official/openibl-init"
FAST_SCNN="/home/leo/usman_ws/datasets/official/fast-scnn/fast_scnn_citys.pth"
DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"


echo ${FILES}


if [ $# -ne 4 ]
  then    
    echo "run ./scripts/train_baseline_dist.sh graphvlad vgg16 pitts 30k"
    exit 1
fi
PORT=6010

#===================================================================================================
# Tiplet Loss
#===================================================================================================

LOSS="triplet"
FILES="/home/leo/usman_ws/models/openibl/fastscnn-grad/${ARCH}-${METHOD}-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
   --method ${METHOD}

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


#===================================================================================================
# SARE Ind Loss
#===================================================================================================
LOSS="sare_ind"
FILES="/home/leo/usman_ws/models/openibl/fastscnn-grad/${ARCH}-${METHOD}-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
   --method ${METHOD}

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

#===================================================================================================
# SARE Joint Loss
#===================================================================================================
LOSS="sare_joint"
FILES="/home/leo/usman_ws/models/openibl/fastscnn-grad/${ARCH}-${METHOD}-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}


$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
   --method ${METHOD}

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

