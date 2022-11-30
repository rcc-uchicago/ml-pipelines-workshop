#!/bin/bash

#Example Midway2 GPU job submission script

#SBATCH --job-name=single-gpu-job 	# Job name (will be displayed in squeue)
#SBATCH --account=pi-[group] 		# Account to be charged
#SBATCH --output=single-gpu-job-%j.out  # Output file name
#SBATCH --error=single-gpu-job-%j.err   # Error file name
#SBATCH --time=24:00:00		        # Job duration (wall time)
#SBATCH --partition=gpu2		# Use GPU partition (note the 2)
#SBATCH --gres=gpu:1			# Request 1 GPU
#SBATCH --cpus-per-task=1		# Number of threads
#SBATCH --ntasks-per-node=1		# Number of CPU cores to drive the GPU

#Load required modules
module load python/anaconda-2021.05
module load cudnn/11.2

#Run your code
python best-network-ever.py
