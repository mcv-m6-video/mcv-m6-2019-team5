#!/usr/bin/env bash
#SBATCH --job-name m6_siamese
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.mcv-m6-2019-team5
#SBATCH --output ../logs/%x_%j.out

source ../venv/bin/activate
python src/train_siamese datasets/siamese_crops/S03_out