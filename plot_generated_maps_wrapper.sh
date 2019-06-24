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

# set the directory for the stored maps 
directory='/Users/juanromero/Documents/Python_3_Files/test'

# The spacing in between each of the labelled x-axis and y-axis ticks
x_tick_spacing=2
y_tick_spacing=2

# The slopes of the power law curves to be plotted
min_power_law_slope=0.5
max_power_law_slope=1.4
power_law_slope_spacing=0.1

# the decimal place to which all power law slopes should be rounded to on legends
legend_decimal_round=1

# store all inputs into an array
inputs[0]=$directory
inputs[1]=$x_tick_spacing
inputs[2]=$y_tick_spacing
inputs[3]=$min_power_law_slope
inputs[4]=$max_power_law_slope
inputs[5]=$power_law_slope_spacing
inputs[6]=$legend_decimal_round

srun -c 1 python plot_maps.py ${inputs[@]}