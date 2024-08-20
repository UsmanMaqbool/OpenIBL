#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.maqboolbhutta@ufl.edu
#SBATCH --time=48:00:00
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
# sbatch --j graphvlad-v7 ./scripts/train_baseline_slurm_all.sh graphvlad pitts
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
TUMPLESIZE=4
CACHEBS=32
NPOCH=10
PORT=6010


INIT_DIR="/home/m.maqboolbhutta/usman_ws/datasets/official/openibl-init"
FAST_SCNN="/home/m.maqboolbhutta/usman_ws/datasets/official/fast-scnn/fast_scnn_citys.pth"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/codes/OpenIBL/examples/data/"




# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/24.1.2 intel/2019.1.144 openmpi/4.0.0
conda activate openibl

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


#===================================================================================================
# SARE Ind Loss
#===================================================================================================
LOSS="sare_ind"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0819-4-fastscnn-agg/${ARCH}-${METHOD}-baseline-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --neg-num 10 --test-batch-size ${CACHEBS} \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs ${NPOCH} --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --method ${METHOD} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
  --num-clusters ${NUMCLUSTER} 


echo "==========Testing============="
FILES="${FILES}/*.tar"
echo ${FILES}
echo "=============================="
for RESUME in $FILES
do
  # take action on each file. $f store current file name
SCALE
  echo "==========################============="
  echo " Testing $RESUME file..."
  echo "======================================="
  $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
   examples/test_pitts_tokyo.py --launcher pytorch \
    -a ${ARCH} --test-batch-size 32 -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME} --fast-scnn=${FAST_SCNN} \
    --num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done


#===================================================================================================
# SARE Joint Loss
#===================================================================================================
LOSS="sare_joint"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0819-4-fastscnn-agg/${ARCH}-${METHOD}-baseline-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --neg-num 10 --test-batch-size ${CACHEBS} \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs ${NPOCH} --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --method ${METHOD} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
  --num-clusters ${NUMCLUSTER}


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
    --resume ${RESUME} --fast-scnn=${FAST_SCNN} \
  --num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done

#===================================================================================================
# Tiplet Loss
#===================================================================================================

LOSS="triplet"
DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/0819-4-fastscnn-agg/${ARCH}-${METHOD}-baseline-${LOSS}-${DATASET}${SCALE}-lr${LR}-tuple${GPUS}-${DATE}"

echo ${FILES}

echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=2 -n${GPUS} \
python -u examples/netvlad_img.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size ${TUMPLESIZE} -j 2 --neg-num 10 --test-batch-size ${CACHEBS} \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs ${NPOCH} --step-size 5 --cache-size 1000 \
  --logs-dir ${FILES} --method ${METHOD} --data-dir ${DATASET_DIR} \
  --init-dir ${INIT_DIR} --fast-scnn=${FAST_SCNN} \
  --num-clusters ${NUMCLUSTER}


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
    --resume ${RESUME} --fast-scnn=${FAST_SCNN} \
  --num-clusters ${NUMCLUSTER}
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done