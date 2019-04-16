#!/usr/bin/env bash
#SBATCH --job-name m6-yolo
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.mcv-m6-2019-team5
#SBATCH --output logs/%x_%j.out

mkdir logs | true
source venv/bin/activate
cd src
python main.py fine_tune_yolo
