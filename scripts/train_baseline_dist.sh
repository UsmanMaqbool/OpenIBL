#!/bin/sh
PYTHON=${PYTHON:-"python3"}
GPUS=1
DATE=$(date '+%d-%b-%H%M') 
DATASET=pitts
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LOSS=$1
LR=0.0001

if [ $# -ne 1 ]
  then
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    exit 1
fi

# while true # find unused tcp port
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
PORT=6010
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 10 --step-size 5 --cache-size 1000 \
  --logs-dir=/media/leo/2C737A9872F69ECF/models/graphnetvlad/${DATASET}${SCALE}-${ARCH}/${LAYERS}-${LOSS}-lr${LR}-tuple${GPUS}-${DATE} 
  #  --resume=/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/netvlad-run/pitts30k-vgg16/conv5-sare_ind-lr0.001-tuple1
/media/leo/2C737A9872F69ECF/models/graphnetvlad/
