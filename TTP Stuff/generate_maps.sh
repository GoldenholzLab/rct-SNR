: '
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
'

monthly_mean_min=1
monthly_mean_max=16
monthly_std_dev_min=1
monthly_std_dev_max=16

#folder='/n/scratch2/jmr95/test_maps_2/cph_power_analysis_parameter_maps'
folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'


inputs[1]=$monthly_mean_min
inputs[2]=$monthly_mean_max
inputs[3]=$monthly_std_dev_min
inputs[4]=$monthly_std_dev_max
inputs[5]=$folder

python generate_maps.py ${inputs[@]}
