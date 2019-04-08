#!/usr/bin/env bash
#SBATCH --job-name m6_siamese
#SBATCH --array=1-3
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.mcv-m6-2019-team5
#SBATCH --output ../logs/%x_%j_%a.out

source venv/bin/activate
echo $SLURM_ARRAY_TASK_ID

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    PATH=datasets/siamese_crops/S01_out
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    PATH=datasets/siamese_crops/S03_out
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
    PATH=datasets/siamese_crops/S04_out
else
   echo "Invalid SLURM_ARRAY_TASK_ID"
   exit 1
fi

python src/train_siamese $PATH