
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

# The names of the files associated with the time-to-prerandomization placebo response map
inputs[8]='expected_TTP_map'
inputs[9]='expected_TTP_map_metadata'
inputs[10]='expected_TTP_plot'
inputs[11]='expected_TTP_plot_with_power_curves'

# The names of the files associated with the histogram of Model 1 patients
inputs[12]='H_model_1_hist' 
inputs[13]='H_model_1_hist_metadata'
inputs[14]='H_model_1_plot'

# The names of the files associated with the histogram of Model 2 patients
inputs[15]='H_model_2_hist' 
inputs[16]='H_model_2_hist_metadata'
inputs[17]='H_model_2_plot'

# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[18]=2
inputs[19]=2

# The slopes of the power law curves to be plotted
inputs[20]=0.5
inputs[21]=1.4
inputs[22]=0.1

python plot_data.py ${inputs[@]}
