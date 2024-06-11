#!/bin/sh
PYTHON=${PYTHON:-"python3"}
ARCH=vgg16

GPUS=1
METHOD="$1"
RESUME="$2"
NUMCLUSTER="64"

PORT=6010

ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
#DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"

echo "==========Testing============="
echo " Testing $RESUME file..."
echo "======================================="
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/test_pitts_tokyo.py --launcher pytorch \
-a ${ARCH} --test-batch-size 32 -j 4 \
--vlad --reduction --method ${METHOD} \
--resume ${RESUME} --esp-encoder ${ESP_ENCODER} \
--num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  