
# The names of the files associated with the 50% responder rate placebo response map
inputs[0]='expected_RR50_map'
inputs[1]='expected_RR50_map_metadata'
inputs[2]='expected_RR50_plot'
inputs[3]='expected_RR50_plot_with_power_curves'

# The names of the files associated with the median percent change placebo response map
inputs[4]='expected_MPC_map'
inputs[5]='expected_MPC_map_metadata'
inputs[6]='expected_MPC_plot'
inputs[7]='expected_MPC_plot_with_power_curves'

# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[8]=2
inputs[9]=2

# The slopes of the power law curves to be plotted
inputs[10]=0.5
inputs[11]=1.4
inputs[12]=0.1

python plot_data.py ${inputs[@]}
