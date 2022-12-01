#!/bin/bash


# Example Midway3 GPU job submission script

#SBATCH --job-name=single-gpu-job 	# Job name (displayed in squeue)
#SBATCH --account=pi-[group]		# Account to be charged
#SBATCH --output=single-gpu-job-%j.out	# Output file name
#SBATCH --error=single-gpu-job-%j.err	# Error file name
#SBATCH --time=10:00:00			# Job duration (wall time)
#SBATCH --partition=gpu			# Use GPU partition (GPU nodes)
#SBATCH --gres=gpu:1			# Request 1 GPU
#SBATCH --constraint=rtx6000		# Only RTX 6000 (can set to v100 or a100)
#SBATCH --cpus-per-task=1		# Number of threads
#SBATCH --ntasks-per-node=1		# Number of CPU cores to drive GPU

# Load required modules
module load python/anaconda-2021.05
module load cudnn/11.2

#Run your code
python best-network-ever.py
