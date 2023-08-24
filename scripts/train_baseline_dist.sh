#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=1
DATE=$(date '+%d-%b') 
DATASET=pitts
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LOSS=$1
LR=0.0001

FILES="/home/leo/usman_ws/models/openibl/official/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${LOSS}-lr${LR}-tuple${GPUS}-${DATE}"
echo ${FILES}

if [ $# -ne 1 ]
  then
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi
PORT=6010
echo "==========Starting Training============="
echo "========================================"
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 2 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES}

echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
  DATASET=pitts
  SCALE=30k
  echo "==========Test on Pitts30k============="
  echo "$RESUME"
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}

   DATASET=tokyo

  echo "==========Test on Tokyo247============="
  echo "$RESUME"
  echo "======================================="


   $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}

done
