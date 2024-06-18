#!/bin/sh
PYTHON=${PYTHON:-"python3"}
GPUS=1
METHOD=graphvlad
DATASET=pitts
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LOSS=sare_ind
LR=0.001
DATE=$(date '+%d-%b')

PORT=6010

FILES="/home/leo/usman_ws/models/openibl/${ARCH}-${METHOD}-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"
INIT_DIR="/home/leo/usman_ws/datasets/openibl-init"
ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"



$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img_sfrs.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size 1 -j 2 --test-batch-size 16 \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 4 --temperature 0.07 0.07 0.06 0.05 --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --esp-encoder=${ESP_ENCODER} \
   --method ${METHOD}
  # --sync-gather
