#!/usr/bin/bash

#SBATCH -c 1                               # 1 core
#SBATCH -t 0-00:10                         # Runtime of 5 minutes, in D-HH:MM format
#SBATCH -p short                           # Run in short partition
#SBATCH -o hostname_%j.out                 # File to which STDOUT + STDERR will be written, including job ID in filename
#SBATCH --mail-type=ALL                    # ALL email notification type
#SBATCH --mail-user=abc123@hms.harvard.edu  # Email to which notifications will be sent

num_iter=50

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

srun -c 1 calculate_RMSE.py 
