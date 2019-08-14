
num_stat_power_estimates=1000
bins=100
#folder='/Users/juanromero/Documents/Python_3_Files/useless_folder'
folder='/n/scratch2/jmr95/TTP_stat_power_estimates'

python -u calculate_RMSE_TTP_power.py $num_stat_power_estimates $bins $folder
