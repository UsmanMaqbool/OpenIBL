#!/bin/sh

PYTHON=${PYTHON:-"python3"}
GPUS=1

# RESUME=$1

ARCH=vgg16


PORT=6010
#triplet
FILES="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-triplet-lr0.0001-tuple1-07-Nov"
#SareIND-off
FILES0="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_ind-lr0.0001-tuple1-11-Nov"
# SARE ind-new
FILES1="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_ind-lr0.0001-tuple1-09-Nov"

#Sarejoint official
FILES2="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.0001-tuple1-08-Nov"

#Sarejoint new
FILES3="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.0001-tuple1-10-Nov"

# FILESALL+=("${FILES}""${FILES1}""${FILES2}")


for RESUME in "${FILES}"/*.tar "${FILES0}"/*.tar "${FILES1}"/*.tar "${FILES2}"/*.tar "${FILES3}"/*.tar
do
  if [ -f "$RESUME" ];then
    echo "Processing $RESUME file..."
    # # take action on each file. $f store current file name
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
  fi
done




# echo ${FILESALL}

# FILES="/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-sare_joint-lr0.001-tuple1-/*.tar"

# echo "==========Testing============="
# echo "=============================="

# for RESUME in $FILES
# do
#   echo "Processing $RESUME file..."
#   # take action on each file. $f store current file name
#   DATASET=pitts
#   SCALE=30k
#   echo "==========Pitts30k============="
#   echo "==============================="
#   $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
#    examples/test.py --launcher pytorch \
#     -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
#     --test-batch-size 32 -j 2 \
#     --vlad --reduction \
#     --resume ${RESUME}

#    DATASET=tokyo

#   echo "==========Tokyo247============="
#   echo "==============================="


#    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
#    examples/test.py --launcher pytorch \
#     -d ${DATASET} --scale ${SCALE} -a ${ARCH} \
#     --test-batch-size 32 -j 2 \
#     --vlad --reduction \
#     --resume ${RESUME}

# done

