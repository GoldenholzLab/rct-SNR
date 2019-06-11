
# The names of the files associated with the 50% responder rate placebo response map
inputs[0]='RR50_stat_power_map'
inputs[1]='RR50_stat_power_map_metadata'
inputs[2]='RR50_stat_power_plot'
inputs[3]='RR50_stat_power_plot_with_power_curves'

# The names of the files associated with the median percent change placebo response map
inputs[4]='MPC_stat_power_map'
inputs[5]='MPC_stat_power_map_metadata'
inputs[6]='MPC_stat_power_plot'
inputs[7]='MPC_stat_power_plot_with_power_curves'

# The names of the files associated with the time-to-prerandomization placebo response map
inputs[8]='TTP_stat_power_map'
inputs[9]='TTP_stat_power_map_metadata'
inputs[10]='TTP_stat_power_plot'
inputs[11]='TTP_stat_power_plot_with_power_curves'

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

# the decimal place to which all power law slopes should be rounded to on legends
inputs[23]=1

python plot_data.py ${inputs[@]}
