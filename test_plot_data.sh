
# The names of the files associated with the 50% responder rate placebo response map
inputs[0]='expected_RR50_map'
inputs[1]='expected_RR50_map_metadata'
inputs[2]='expected_RR50_plot'

# The names of the files associated with the median percent change placebo response map
inputs[3]='expected_MPC_map'
inputs[4]='expected_MPC_map_metadata'
inputs[5]='expected_MPC_plot'

# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[6]=1
inputs[7]=1

# The slopes of the power law curves to be plotted
inputs[8]=0.5
inputs[9]=1.4
inputs[10]=0.1

python plot_data.py ${inputs[@]}
