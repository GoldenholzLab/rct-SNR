#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-02:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate main_env

inputs[0]=0
inputs[1]=16
inputs[2]=0.1
inputs[3]=0
inputs[4]=16
inputs[5]=0.1
inputs[6]=5
inputs[7]=153
inputs[8]=2
inputs[9]=3
inputs[10]=0
inputs[11]=0.05
inputs[12]=0.2
inputs[13]=0.05
inputs[14]=3
inputs[15]='runtime_data'

srun -c 1 python generate_voxels.py ${inputs[@]}
