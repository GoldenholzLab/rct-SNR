import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_expected_endpoint_maps_and_models(monthly_mu_axis_start, monthly_mu_axis_stop, monthly_mu_axis_step,
                                           monthly_sigma_axis_start, monthly_sigma_axis_stop, monthly_sigma_axis_step,
                                           num_monthly_mu, monthly_mu_array, monthly_mu_tick_spacing,
                                           num_monthly_sigma, monthly_sigma_tick_spacing,
                                           Model_1_expected_RR50, Model_1_expected_MPC, 
                                           Model_2_expected_RR50, Model_2_expected_MPC, 
                                           expected_RR50_map, expected_MPC_map, H_model_1, H_model_2,
                                           expected_RR50_filename, expected_MPC_filename,
                                           expected_RR50_with_curves_filename,expected_MPC_with_curves_filename,
                                           model_1_2D_hist_filename, model_2_2D_hist_filename,
                                           min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
                                           legend_decimal_round):
    
    # generate the tick labels for the monthly mean axis as well as for the monthly standard deviation axis
    # the tick labels are distinctly different from the actual ticks
    monthly_mu_tick_labels = np.arange(monthly_mu_axis_start, monthly_mu_axis_stop + monthly_mu_tick_spacing, monthly_mu_tick_spacing)
    monthly_sigma_tick_labels = np.arange(monthly_sigma_axis_start, monthly_sigma_axis_stop + monthly_sigma_tick_spacing, monthly_sigma_tick_spacing)

    # get the number of tick labels for both axes
    num_monthly_mu_tick_labels = len(monthly_mu_tick_labels)
    num_monthly_sigma_tick_labels = len(monthly_sigma_tick_labels)

    # calculate the ticks (location of each tick) for the monthly mean axis and monthly standard deviation axis
    monthly_mu_ticks = monthly_mu_tick_labels/monthly_mu_axis_step + 0.5*np.ones(num_monthly_mu_tick_labels)
    monthly_sigma_ticks = np.flip( monthly_sigma_tick_labels/monthly_sigma_axis_step + 0.5*np.ones(num_monthly_sigma_tick_labels), 0)

    # plot the expected RR50
    plt.figure()
    ax = sns.heatmap(expected_RR50_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected RR50 Placebo')
    plt.savefig(os.getcwd() + '/' + expected_RR50_filename + '.png')

    # plot the expected MPC
    plt.figure()
    ax = sns.heatmap(expected_MPC_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected MPC Placebo')
    plt.savefig(os.getcwd() + '/' + expected_MPC_filename + '.png')

    # get the scaling ratios for any other data that needs to be plotted on top of the heatmaps
    monthly_mu_scale_ratio = num_monthly_mu/(monthly_mu_axis_stop - monthly_mu_axis_start)
    monthly_sigma_scale_ratio = num_monthly_sigma/(monthly_sigma_axis_stop - monthly_sigma_axis_start)
    power_law_slopes = np.arange(min_power_law_slope/power_law_slope_spacing, max_power_law_slope/power_law_slope_spacing + 1, 1)*power_law_slope_spacing

    # plot the expected RR50 with corresponding power law curves
    plt.figure()
    ax = sns.heatmap(expected_RR50_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    for power_law_slope_index in range(len(power_law_slopes)):
        power_law_slope = power_law_slopes[power_law_slope_index]
        power_law_monthly_sigma_array = np.power(monthly_mu_array, power_law_slope)
        plt.plot(monthly_mu_array*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - power_law_monthly_sigma_array)*monthly_sigma_scale_ratio )
    plt.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] )
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected RR50 Placebo')
    plt.savefig(os.getcwd() + '/' + expected_RR50_with_curves_filename + '.png')

    # plot the expected MPC with corresponding power law curves
    plt.figure()
    ax = sns.heatmap(expected_MPC_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    for power_law_slope_index in range(len(power_law_slopes)):
        power_law_slope = power_law_slopes[power_law_slope_index]
        power_law_monthly_sigma_array = np.power(monthly_mu_array, power_law_slope)
        plt.plot(monthly_mu_array*monthly_mu_scale_ratio, (monthly_sigma_axis_stop - power_law_monthly_sigma_array)*monthly_sigma_scale_ratio )
    plt.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] ) 
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Expected MPC Placebo')
    plt.savefig(os.getcwd() + '/' + expected_MPC_with_curves_filename + '.png')

    # plot the 2D histogram of Model 1 patients
    plt.figure()
    ax = sns.heatmap(H_model_1, cbar_kws={'label':'probability of sampling patient from a point'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Model 1 patient population')
    plt.savefig(os.getcwd() + '/' + model_1_2D_hist_filename + '.png')

    # plot the 2D histogram of Model 2 patients
    plt.figure()
    ax = sns.heatmap(H_model_2, cbar_kws={'label':'probability of sampling patient from a point'})
    ax.set_xticks(monthly_mu_ticks)
    ax.set_xticklabels(monthly_mu_tick_labels, rotation='horizontal')
    ax.set_yticks(monthly_sigma_ticks)
    ax.set_yticklabels(monthly_sigma_tick_labels, rotation='horizontal')
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title('Model 2 patient population')
    plt.savefig(os.getcwd() + '/' + model_2_2D_hist_filename + '.png')

