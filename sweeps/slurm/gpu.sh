#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00
#SBATCH --mem=65536
#SBATCH --cpus-per-task=4
#SBATCH -o "/nfs/students/borchero/logs/job-%A_%a.out"

echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate base

cd /nfs/students/borchero/natpn

# Read the appropriate line in the input file
OPTIONS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $1)

# And run the training job
poetry run train --experiment slurm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} ${OPTIONS}
