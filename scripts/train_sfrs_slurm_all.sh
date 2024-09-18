#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.maqboolbhutta@ufl.edu
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=8GB
#SBATCH --distribution=cyclic:cyclic

## To RUN
# sbatch --j 0904-s1 scripts/leo/train_sfrs_slurm_all.sh graphvlad vgg16 pitts 30k
####################################################################################################

# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment
PYTHON=${PYTHON:-"python"}
allowed_arguments_list1=("netvlad" "graphvlad")
allowed_arguments_list2=("triplet" "sare_ind" "sare_joint")

if [ "$#" -lt 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <DATASET (pitts|tokyo)>"
    echo "./train_baseline_dist.sh netvlad triplet pitts"    
    exit 1
fi

GPUS=4
METHOD="$1"
ARCH="$2"
DATASET="$3"
SCALE="$4"
NUMCLUSTER=64
LAYERS=conv5
LR=0.001
TUMPLESIZE=1
CACHEBS=16
NPOCH=5
PORT=6010


INIT_DIR="/blue/hmedeiros/m.maqboolbhutta/datasets/official/openibl-init"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/codes/OpenIBL/examples/data/"
FAST_SCNN="/home/m.maqboolbhutta/usman_ws/datasets/official/fast_scnn/fast_scnn_citys.pth"


# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/24.3.0 intel/2019.1.144 openmpi/4.0.0
conda activate openibl3

# PRINTS
#=======
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST" 
echo "Other nodes: $NODES"



### Create cluster
# python -u examples/cluster.py -d pitts -a ${ARCH} -b 64 --num-clusters ${NUMCLUSTER} \
# --width 640 --height 480 --data-dir ${DATASET_DIR} \
# --init-dir ${INIT_DIR}


# ===================================================================================================
#SARE Ind Loss
# ===================================================================================================
LOSS="sare_ind"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0915-try1/${ARCH}-${METHOD}_SFRS-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img_sfrs.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --test-batch-size ${CACHEBS} \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 4 --temperature 0.07 0.07 0.06 0.05 --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
  --method ${METHOD} 

echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
  echo "==========################============="
  echo " Testing $RESUME file on Pitts250k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 250k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts250k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Pitts30k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 30k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts30k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Tokyo..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d tokyo
  echo "==========################============="
  echo " Done Testing with $RESUME file on Tokyo..."
  echo "======================================="  
done


#===================================================================================================
# SARE Joint Loss
#===================================================================================================
LOSS="sare_joint"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0915-try1/${ARCH}-${METHOD}_SFRS-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img_sfrs.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --test-batch-size ${CACHEBS} \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 4 --temperature 0.07 0.07 0.06 0.05 --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
  --method ${METHOD}


echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
  echo "==========################============="
  echo " Testing $RESUME file on Pitts250k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 250k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts250k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Pitts30k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 30k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts30k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Tokyo..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d tokyo
  echo "==========################============="
  echo " Done Testing with $RESUME file on Tokyo..."
  echo "======================================="  
done

#===================================================================================================
# Tiplet Loss
#===================================================================================================

LOSS="triplet"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0915-try1/${ARCH}-${METHOD}_SFRS-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img_sfrs.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --syncbn \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --test-batch-size ${CACHEBS} \
  --neg-num 10  --pos-pool 20 --neg-pool 1000 --pos-num 10 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} --soft-weight 0.5 \
  --eval-step 1 --epochs 5 --step-size 5 --cache-size 1000 --generations 4 --temperature 0.07 0.07 0.06 0.05 --logs-dir ${FILES} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
   --method ${METHOD}


echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
  echo "==========################============="
  echo " Testing $RESUME file on Pitts250k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 250k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts250k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Pitts30k..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d pitts --scale 30k
  echo "==========################============="
  echo " Done Testing with $RESUME file on Pitts30k..."
  echo "======================================="  

  echo "==========################============="
  echo " Testing $RESUME file on Tokyo..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test.py --launcher pytorch \
    -a ${ARCH} --test-batch-size ${CACHEBS} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d tokyo
  echo "==========################============="
  echo " Done Testing with $RESUME file on Tokyo..."
  echo "======================================="  
done