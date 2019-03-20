#!/usr/bin/env bash
#SBATCH --job-name fcn8_camvid
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --qos masterhigh
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/mcv-m6-2019-team5
#SBATCH --output logs/%x_%j.out

mkdir logs | true
source venv/bin/activate
python src/main.py fine_tune_yolo
