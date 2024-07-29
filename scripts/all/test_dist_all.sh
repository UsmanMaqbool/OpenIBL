#!/bin/sh
PYTHON=${PYTHON:-"python3"}
ARCH=vgg16

GPUS=1
METHOD="$1"
FILES="$2"
NUMCLUSTER="64"

PORT=6010

ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
#DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"

echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
  echo "==========################============="
  echo " Testing $RESUME file..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test_pitts_tokyo.py --launcher pytorch \
    -a ${ARCH} --test-batch-size 32 -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME} --esp-encoder ${ESP_ENCODER} \
    --num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done