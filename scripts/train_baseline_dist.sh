#!/bin/sh
PYTHON=${PYTHON:-"python"}
GPUS=1
METHOD=$1
LOSS=$2
DATASET=$3
SCALE=30k
ARCH=vgg16
LAYERS=conv5
LR=0.001

DATE=$(date '+%d-%b') 
FILES="/home/leo/usman_ws/models/openibl/${DATASET}-${METHOD}-${LOSS}-lr${LR}-${DATE}"

DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"
INIT_DIR="/home/leo/usman_ws/datasets/openibl-init"
ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"


if [ "$#" -lt 3 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "Arguments error: <DATASET (pitts|tokyo)>"
    echo "./train_baseline_dist.sh netvlad triplet pitts"    
    exit 1
fi


echo "========================================"
echo "saving checkpoints at ${FILES}"
echo "========================================"


PORT=6010
echo "==========Starting Training============="
echo "========================================"
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
examples/netvlad_img.py --launcher pytorch --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 1 -j 1 --neg-num 10 --test-batch-size 32 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --method ${METHOD} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --esp_encoder=${ESP_ENCODER} 


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
    -a ${ARCH} --test-batch-size 32 -j 4 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME} --esp-encoder ${ESP_ENCODER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done

done
