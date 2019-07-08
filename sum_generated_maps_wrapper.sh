#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -t 0-00:10
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

directory='/n/scratch2/jmr95/test_maps_2'
shape_1=24.143
scale_1=297.366
alpha_1=284.024
beta_1=369.628
shape_2=111.313
scale_2=296.728
alpha_2=296.339
beta_2=243.719

inputs[0]=$directory
inputs[1]=$shape_1
inputs[3]=$scale_1
inputs[4]=$alpha_1
inputs[5]=$beta_1
inputs[6]=$shape_2
inputs[7]=$scale_2
inputs[8]=$alpha_2
inputs[9]=$beta_2

srun -c 1 python sum_generated_maps.py ${inputs[@]}}