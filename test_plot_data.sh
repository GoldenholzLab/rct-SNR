
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

# The names of the files associated with the histogram of Model 1 patients
inputs[8]='H_model_1_hist' 
inputs[9]='H_model_1_hist_metadata'
inputs[10]='H_model_1_plot'

# The names of the files associated with the histogram of Model 2 patients
inputs[11]='H_model_2_hist' 
inputs[12]='H_model_2_hist_metadata'
inputs[13]='H_model_2_plot'

# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[14]=2
inputs[15]=2

# The slopes of the power law curves to be plotted
inputs[16]=0.5
inputs[17]=1.4
inputs[18]=0.1

python plot_data.py ${inputs[@]}
