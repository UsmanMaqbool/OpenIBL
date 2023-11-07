#!/bin/sh

PYTHON=${PYTHON:-"python3"}
GPUS=1

# RESUME=$1

ARCH=vgg16


PORT=6010

FILES="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_ind-lr0.0001-tuple1-05-Sep/*.tar"

echo "==========Testing============="
echo "=============================="

for RESUME in $FILES
do
  echo "Processing $RESUME file..."
  # take action on each file. $f store current file name
  DATASET=pitts
  SCALE=30k
  echo "==========Pitts30k============="
  echo "==============================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}

   DATASET=tokyo

  echo "==========Tokyo247============="
  echo "==============================="


   $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}

done



# $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
# examples/test.py --launcher pytorch \
#     -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
#     --test-batch-size 32 -j 2 \
#     --vlad --reduction \
#     --resume ${RESUME}
#     # --sync-gather
