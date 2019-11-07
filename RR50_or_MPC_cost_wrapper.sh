
#!/usr/bin/bash

#SBATCH -p short
#SBATCH --mem-per-cpu=10G
#SBATCH -t 0-01:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e jmr95_%j.err
#SBATCH -o jmr95_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jromero5@bidmc.harvard.edu

inputs[0]=$1
inputs[1]=$2
inputs[2]=$3

module load gcc/6.2.0
module load conda2/4.2.13
module load python/3.6.0
source activate working_env

srun -c 1 python -u main_python_scripts/RR50_or_MPC_cost.py ${inputs[@]}

: ' 
inputs[0]=$1
inputs[1]=$2
inputs[2]=$3

python main_python_scripts/RR50_or_MPC_cost.py ${inputs[@]}
'