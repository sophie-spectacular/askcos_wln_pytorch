#!/bin/bash
#SBATCH -p long                 # select partition
#SBATCH -J test-torch-1                 # job name
#SBATCH -t 99-23               # time requested HH:MM:SS or DD-HH
#SBATCH -n 2                      # number of tasks (cpus), default 1 max 56
#SBATCH --mem=10G               # request memory, default 3 GB max 90112(MB)
#SBATCH --gres=gpu:1              # request gpu resources (askcosgpu01 only)
#SBATCH --mail-type=END,FAIL      # types of emails to get
#SBATCH --mail-user=sophiedh@mit.edu  # email address
############################################################
# ADDITIONAL NOTES (this section can be discarded)
#
# Max time and priority for each partition
#   normal:    14 days  higher priority
#   long:    unlimited  lower priority
#
# If you want email notifications, you can set --mail-type
# and --mail-user. Useful values for --mail-type:
#   BEGIN, END, FAIL, REQUEUE, ALL
############################################################
#
echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

python train_all.py --nproc 4 --train data/train.txt.proc --valid data/valid.txt.proc --test data/test.txt.proc --model-name uspto_500k --model-dir uspto_500k --epochs 1 --cutoff 0.0
