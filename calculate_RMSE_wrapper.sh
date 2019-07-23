#!/usr/bin/bash

#SBATCH -c 1                               
#SBATCH -t 0-00:10                         
#SBATCH -p short                           
#SBATCH -o jmr95_%j.out
#SBATCH -e jmr95_%j.out             
#SBATCH --mail-type=ALL                    
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

num_iter=50

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

srun -c 1 calculate_RMSE.py 
