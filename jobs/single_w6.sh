#!/usr/bin/env bash
#SBATCH --job-name m6_siamese
#SBATCH --array=0-17
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.mcv-m6-2019-team5
#SBATCH --output ../logs/%x_%A_%a.out

source venv/bin/activate

if [[ $(($SLURM_ARRAY_TASK_ID / 9)) -eq 0 ]]; then
    TRACKING_TYPE=single
elif [[ $(($SLURM_ARRAY_TASK_ID / 9)) -eq 1 ]]; then
    TRACKING_TYPE=multiple
fi

if [[ $(($SLURM_ARRAY_TASK_ID % 3)) -eq 0 ]]; then
    SEQUENCE=train_seq1
elif [[ $(($SLURM_ARRAY_TASK_ID % 3))  -eq 1 ]]; then
    SEQUENCE=train_seq3
elif [[ $(($SLURM_ARRAY_TASK_ID % 3))  -eq 2 ]]; then
    SEQUENCE=train_seq4
fi

if [[ $(($SLURM_ARRAY_TASK_ID / 3 % 3)) -eq 0 ]]; then
    METHOD=kalman
elif [[ $(($SLURM_ARRAY_TASK_ID / 3 % 3))  -eq 1 ]]; then
    METHOD=overlap
elif [[ $(($SLURM_ARRAY_TASK_ID / 3 % 3))  -eq 2 ]]; then
    METHOD=optical_flow
fi

cd src
echo ${TRACKING_TYPE} ${SEQUENCE} ${METHOD}
python main.py ${TRACKING_TYPE} ${SEQUENCE} ${METHOD}