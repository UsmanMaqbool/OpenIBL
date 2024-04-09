#!/bin/sh

PYTHON=${PYTHON:-"python3"}
GPUS=1

# RESUME=$1

ARCH=vgg16
DATASET_DIR="/home/leo/usman_ws/codes/OpenIBL/examples/data/"
INIT_DIR="/home/leo/usman_ws/datasets/openibl-init"
ESP_ENCODER="/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
METHOD="graphvlad"
PORT=6010
#triplet
FILES="/home/leo/usman_ws/models/openibl/hipergator/6-Apr"
# #SareIND-off
# FILES0="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_ind-lr0.0001-tuple1-11-Nov"
# # SARE ind-new
# FILES1="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_ind-lr0.0001-tuple1-09-Nov"

# #Sarejoint official
# FILES2="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.0001-tuple1-08-Nov"

# #Sarejoint new
# FILES3="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.0001-tuple1-10-Nov"



# for RESUME in "${FILES}"/*.tar "${FILES0}"/*.tar "${FILES1}"/*.tar "${FILES2}"/*.tar "${FILES3}"/*.tar
for RESUME in $(find "${FILES}" -name "*.tar")

do
  # take action on each file. $f store current file name
  echo "==========################============="
  echo " Testing $RESUME file..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    examples/test_pitts_tokyo.py --launcher pytorch \
    -a ${ARCH} --test-batch-size 20 -j 4 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME} --esp-encoder ${ESP_ENCODER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done

