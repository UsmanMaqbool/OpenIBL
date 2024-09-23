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
#SBATCH --gpus-per-node=a100:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --constraint=a100
#SBATCH --mem-per-cpu=8GB
#SBATCH --distribution=cyclic:cyclic

## To RUN
#  sbatch --j 0921 scripts/leo/test_slurm_all.sh graphvlad vgg16 /home/m.maqboolbhutta/usman_ws/models/openibl/0921-try4/vgg16-graphvlad_SFRS-sare_ind-pitts30k-lr0.001-tuple4-21-Sep/checkpoint2_4.pth.tar
####################################################################################################

# PYTHON SCRIPT
#==============

#This is the python script to run in the pytorch environment
PYTHON=${PYTHON:-"python"}
allowed_arguments_list1=("netvlad" "graphvlad")

if [ "$#" -lt 2 ]; then
    echo "Arguments error: <METHOD (netvlad|graphvlad>"
    echo "./train_baseline_dist.sh netvlad triplet pitts"    
    exit 1
fi

GPUS=4
METHOD="$1"
ARCH="$2"
FILES="$3"
NUMCLUSTER=64
LAYERS=conv5
CACHEBS=16
PORT=6010



INIT_DIR="/home/m.maqboolbhutta/usman_ws/datasets/official/openibl-init"
FAST_SCNN="/home/m.maqboolbhutta/usman_ws/datasets/official/fast_scnn/fast_scnn_citys.pth"
DATASET_DIR="/home/m.maqboolbhutta/usman_ws/codes/OpenIBL/examples/data/"



# LOAD PYTORCH SOFTWARE ENVIRONMENT
#==================================

## You can load a software environment or use a singularity container.
## CONTAINER="singularity exec --nv /path/to/container.sif" (--nv option is to enable gpu)
module purge
module load conda/24.3.0 intel/2019.1.144 openmpi/4.0.0
conda activate openibl

# PRINTS
#=======
date; pwd; which python
export HOST=$(hostname -s)
NODES=$(scontrol show hostnames | grep -v $HOST | tr '\n' ' ')
echo "Host: $HOST" 
echo "Other nodes: $NODES"



echo "==========Testing============="
if [ -d "${FILES}" ]; then
  FILES="${FILES}/*.tar"
else
  FILES="${FILES}"
fi
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
    -a ${ARCH} -j 2 \
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
    -a ${ARCH} -j 2 \
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
    -a ${ARCH} -j 2 \
    --vlad --reduction --method ${METHOD} \
    --resume ${RESUME}  --fast-scnn=${FAST_SCNN}  \
    --num-clusters ${NUMCLUSTER} -d tokyo
  echo "==========################============="
  echo " Done Testing with $RESUME file on Tokyo..."
  echo "======================================="  
done

