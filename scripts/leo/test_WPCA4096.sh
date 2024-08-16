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
FILES="${FILES}/*WPCA4096.pth.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  echo "==========################============="
  echo " Testing $RESUME file on Pitts250k ..."
  echo "======================================="
  DATASET=pitts
  SCALE=250k
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test-model-with-pca.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH}\
    --test-batch-size 24 -j 2 \
    --vlad --method ${METHOD} \
    --resume ${RESUME} --fast-scnn ${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}

  echo "==========################============="
  echo " Testing $RESUME file on Pitts30k ..."
  echo "======================================="
  DATASET=pitts
  SCALE=30k
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test-model-with-pca.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH}\
    --test-batch-size 24 -j 2 \
    --vlad --method ${METHOD} \
    --resume ${RESUME} --fast-scnn ${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}

  echo "==========################============="
  echo " Testing $RESUME file on Tokyo24/7 ..."
  echo "======================================="
  DATASET=tokyo
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test-model-with-pca.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH}\
    --test-batch-size 24 -j 2 \
    --vlad --method ${METHOD} \
    --resume ${RESUME} --fast-scnn ${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}  
 

  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done