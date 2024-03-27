#!/bin/bash
# Launch pytorch distributed in a software environment or container
#
# (c) 2022, Eric Stubbs
# University of Florida Research Computing

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.maqboolbhutta@ufl.edu
#SBATCH --time=52:00:00
#SBATCH --partition=gpu
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=a100:8   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=4GB
#SBATCH --distribution=cyclic:cyclic





# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment
PYTHON=${PYTHON:-"python"}
allowed_arguments_list1=("netvlad" "graphvlad")
allowed_arguments_list2=("triplet" "sare_ind" "sare_joint")

if [ "$#" -lt 3 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "Arguments error: <LOSS_TYPE (triplet|sare_ind|sare_joint)>"
    echo "Arguments error: <DATASET (pitts|tokyo)>"
    echo "./train_baseline_dist.sh netvlad triplet pitts"    
    exit 1
fi

METHOD="$1"
LOSS="$2"
DATASET=$3
DATE=$(date '+%d-%b')



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

DATE=$(date '+%d-%b') 
FILES="/home/m.maqboolbhutta/usman_ws/models/openibl/${DATASET}-${METHOD}-${LOSS}-lr${LR}-${DATE}"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/codes/OpenIBL/examples/data/"
INIT_DIR="/blue/hmedeiros/m.maqboolbhutta/datasets/openibl-init"
ESP_ENCODER="/home/m.maqboolbhutta/usman_ws/datasets/netvlad-official/espnet-encoder/espnet_p_2_q_8.pth"

echo ${FILES}



GPUS=8

SCALE=30k
ARCH=vgg16
LAYERS=conv5
LR=0.001

PORT=6010
echo "==========Starting Training============="
echo "========================================"
srun --mpi=pmix_v3 -p=gpu --cpus-per-task=4 -n${GPUS} \
python -u examples/netvlad_img.py --launcher slurm --tcp-port ${PORT} \
  -d ${DATASET} --scale ${SCALE} \
  -a ${ARCH} --layers ${LAYERS} --vlad --syncbn --sync-gather \
  --width 640 --height 480 --tuple-size 5 -j 4 --neg-num 10 --test-batch-size 24 \
  --margin 0.1 --lr ${LR} --weight-decay 0.001 --loss-type ${LOSS} \
  --eval-step 1 --epochs 10 --step-size 5 --cache-size 500 \
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
    --resume ${RESUME} --esp_encoder=${ESP_ENCODER} \
  echo "==========################============="
  echo " Done Testing with $RESUME file..."
  echo "======================================="  
done