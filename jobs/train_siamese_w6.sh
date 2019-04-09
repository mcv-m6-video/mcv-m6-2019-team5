#!/usr/bin/env bash
#SBATCH --job-name m6_siamese
#SBATCH --array=1-4
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:Pascal:1
#SBATCH --chdir /home/grupo06/.mcv-m6-2019-team5
#SBATCH --output ../logs/%x_%A_%a.out

source venv/bin/activate

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    DATASET_PATH=../datasets/siamese_crops/S01_out
    VALIDATION_PATH=../datasets/siamese_crops/S01
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    DATASET_PATH=../datasets/siamese_crops/S03_out
    VALIDATION_PATH=../datasets/siamese_crops/S03
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
    DATASET_PATH=../datasets/siamese_crops/S04_out
    VALIDATION_PATH=../datasets/siamese_crops/S04
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then
    DATASET_PATH=../datasets/siamese_crops/all
else
   echo "Invalid SLURM_ARRAY_TASK_ID"
   exit 1
fi

cd src
echo ${DATASET_PATH}
echo ${VALIDATION_PATH}
python train_siamese.py ${DATASET_PATH} ${VALIDATION_PATH}