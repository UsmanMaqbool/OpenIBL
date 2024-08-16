#!/bin/sh
PYTHON=${PYTHON:-"python3"}
ARCH=vgg16

GPUS=1
METHOD="$1"
FILES="$2"
NUMCLUSTER="64"
PORT=6010

FAST_SCNN="/home/leo/usman_ws/datasets/official/fast-scnn/fast_scnn_citys.pth"

echo "==========Testing============="
FILES=$(ls -r ${FILES}/*.tar)
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
    --resume ${RESUME} --fast-scnn ${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done