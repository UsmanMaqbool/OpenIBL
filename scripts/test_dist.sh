#!/bin/sh
PYTHON=${PYTHON:-"python3"}
GPUS=1

RESUME=$1
ARCH=vgg16

DATASET=${2-pitts}
SCALE=${3-250k}

PORT=6010

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/test.py --launcher pytorch \
    -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
    --test-batch-size 32 -j 2 \
    --vlad --reduction \
    --resume ${RESUME}
    # --sync-gather
