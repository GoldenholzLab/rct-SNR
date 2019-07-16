#!/usr/bin/bash

#SBATCH -p short
#SBATCH -t 0-00:03
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o jmr95_%j.out
#SBATCH -e jmr95_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

for ((i=1; i<16; i=i+1));
do
    sbatch test_Fisher_Exact.sh $i
done