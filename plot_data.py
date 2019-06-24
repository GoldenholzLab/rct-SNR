import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

'''
def retrieve_map(data_map_file_name):


    Purpose:

        This function retrives a specified data map from the file system along with

        the metadata needed to help plot that data map, both of which are assumed to

        be stored in JSON files located in the same folder as this python script.

    Inputs:

        1) data_map_file_name:

            (string) - the name of the JSON file containing the actual data map
        
    Outputs:

        1) data_map:

            (2D Numpy array) - the 2D map of data to be plotted

        2) x_axis_start:

            (float) - the beginning value of the x-axis for the plot of the data map

        3) x_axis_stop:

            (float) - the end value of the x-axis for the plot of the data map

        4) x_axis_step:

            (float) - the size of the spaces in between each x-axis value 

        5) y_axis_start:

            (float) - the beginning value of the y-axis for the plot of the data map

        6) y_axis_stop:

            (float) - the end value of the y-axis for the plot of the data map

        7) y_axis_step:

            (float) - the size of the spaces in between each y-axis value



    # create the string representation of the file path for the data map
    data_map_file_path = os.getcwd() + '/' + data_map_file_name + '.json'

    # open the data map file
    with open(data_map_file_path, 'r') as map_storage_file:

        # load the data map into memory
        data_map = np.array(json.load(map_storage_file))
    
    return [data_map]
'''


def plot_power_law_curves(ax, min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
                          x_axis_start, x_axis_stop, x_axis_step, 
                          y_axis_start, y_axis_stop, y_axis_step,
                          legend_decimal_round):
    '''

    Purpose:

        This function plots the power law curves onto a heatmap which has already been plotted.

    Inputs:

        1) ax:

            (matplotplib.pyplot.axis object) - the axis object corresponding to the heatmap which should 
                                               
                                               have already been plotted.

        2) min_power_law_slope:

            (float) - the minimum power law slope to be plotted

        3) max_power_law_slope:

            (float) - the maximum power law slope to be plotted

        4) power_law_slope_spacing:

            (float) - the spacing in between each power law slope

        5) x_axis_start:

            (float) - the beginning value of the x-axis for the plot of the heatmap

        6) x_axis_stop:

            (float) - the end value of the x-axis for the plot of the heatmap
        
        7) x_axis_step:

            (float) - the size of the spaces in between each x-axis value

        8) y_axis_start:

            (float) - the beginning value of the y-axis for the plot of the heatmap

        9) y_axis_stop:

            (float) - the end value of the y-axis for the plot of the heatmap

        10) y_axis_step:

            (float) - the size of the spaces in between each y-axis value
        
        11) legend_decimal_round:

            (int) - the decimal place to which the power law slopes should be rounded to
                    
                    within the legend of power law slopes
    
    Outputs:

        1) ax:

            (matplotlib.axes.Axes object) - the axes object containing the plot of the map with
                                            
                                            power law curves and an accompanying legend 

    '''

    # calculate the scaling ratios for plotting the power law curves on the heatmaps
    x_axis_scale_ratio = (1/x_axis_step) + (1/(x_axis_stop - x_axis_start))
    y_axis_scale_ratio = (1/y_axis_step) + (1/(y_axis_stop - y_axis_start))

    # recalculate the x-axis array for the purpose of giving the power law curves something to be plotted against
    x_axis_array = np.arange(x_axis_start, x_axis_stop + x_axis_step, x_axis_step)

    # calculate the array of power law slopes
    power_law_slopes = np.arange(min_power_law_slope/power_law_slope_spacing, max_power_law_slope/power_law_slope_spacing + 1, 1)*power_law_slope_spacing

    # for each power law slope
    for power_law_slope_index in range(len(power_law_slopes)):

        # retrieve the relevant power law slope from the power-law-slope array
        power_law_slope = power_law_slopes[power_law_slope_index]

        # calculate the points on the power law curve to be plotted for the relevant power law slope
        power_law_y_axis_points = np.power(x_axis_array, power_law_slope)

        # actaully plot the power law curve corresponding to this power law slope
        ax.plot(x_axis_array*x_axis_scale_ratio, (y_axis_stop - power_law_y_axis_points)*y_axis_scale_ratio )

    # add the legend saying what slope each curve corresponds to
    ax.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] ) 

    return ax


def plot_map(data_map_file_name, data_plot_file_name,
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, plot_curves,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             map_title, legend_decimal_round):
    '''

    Purpose:

        This function plots one statistical power map with the given input parameters.

    Inputs:
    
        1) data_map_file_name:

            (string) - the name of the file containing the data map to be plotted

        2) data_map_metadata_file_name:

            (string) - the name of the file containing the metadata for the data map to be plotted

        3) data_plot_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the data map

        4) x_tick_spacing

            (float) - the spacing in between each of the labelled x-axis ticks

        5) y_tick_spacing:

            (float) - the spacing in between each of the labelled y-axis ticks

        6) plot_curves

            (boolean) - a boolean flag, which if positive, will plot the power law curves on this
                        
                        data map 

        7) min_power_law_slope:

            (float) - the minimum power law slope to be plotted
        
        8) max_power_law_slope:

            (float) - the maximum power law slope to be plotted
        
        9) power_law_slope_spacing:

            (float) - the size of the spaces in between each y-axis value

        10) map_title:

            (string) - the title of the map to be plotted
        
        11) legend_decimal_round:

            (int) - the decimal place to which the power law slopes should be rounded to
                    
                    within the legends of power law slopes

    Outputs:

        Technically None

    '''

    # create the string representation of the folder in which the data map is stored
    data_map_folder = directory + '/' + str(min_req_base_sz_count) + '/final'

     # create the string representation of the file path for the data map
    data_map_file_path = data_map_folder + '/' + data_map_file_name + '.json'

    # open the data map file
    with open(data_map_file_path, 'r') as map_storage_file:

        # load the data map into memory
        data_map = np.array(json.load(map_storage_file))

    # figure out the file path of the PNG picture of the data to be plotted
    data_plot_file_path = os.getcwd() + '/' + data_plot_file_name + '.png'

    # figure out the spacing of the tick labels for both axes
    x_tick_labels = np.arange(x_axis_start, x_axis_stop + x_tick_spacing, x_tick_spacing)
    y_tick_labels = np.arange(y_axis_start, y_axis_stop + y_tick_spacing, y_tick_spacing)

    # figure out the number of tick lables for both axes
    num_x_tick_labels = len(x_tick_labels)
    num_y_tick_labels = len(y_tick_labels)

    # figure out the actual ticks for both labels
    x_ticks = x_tick_labels/x_axis_step + 0.5*np.ones(num_x_tick_labels)
    y_ticks = y_tick_labels/y_axis_step + 0.5*np.ones(num_y_tick_labels)

    # flip the y axis ticks
    y_ticks = np.flip(y_ticks, 0)

    # create a figure
    fig = plt.figure()

    # plot the map
    ax = sns.heatmap(data_map, cbar_kws={'label':'estimated expected endpoint response in percentages'})
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title(map_title)

    # make the x-axis labels make sense
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation='horizontal')

    # make the y-axis labels make sense
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, rotation='horizontal')

    # if told to plot the power law curves:
    if(plot_curves):

        # plot the power law curves
        ax = plot_power_law_curves(ax, min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
                                   x_axis_start, x_axis_stop, x_axis_step, 
                                   y_axis_start, y_axis_stop, y_axis_step,
                                   legend_decimal_round)

    # save the figure as a PNG file in the same folder as this script
    fig.savefig( data_plot_file_path )

    # close the figure
    plt.close(fig)

'''
def plot_histogram(H_model_file_name, H_model_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_plot_title):

    Purpose:

        This function plots one histogram of all the patients generated from one NV model.

    Inputs:

        1) H_model_file_name:
        
            (string) - the name of the file containing the histogram to be plotted

        2) H_model_metadata_file_name:
        
            (string) - the name of the file containing the metadata for the histogram to be plotted

        3) H_model_plot_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the 

                       histogram of the NV model patients

        4) x_tick_spacing:
        
            (float) - the spacing in between each of the labelled x-axis ticks

        5) y_tick_spacing:
        
            (float) - the spacing in between each of the labelled y-axis ticks

        6) H_model_plot_title:

            (string) - the title of the histogram to be plotted

    Outputs:

        Technically None


    # retrieve the histogram of the NV model and its corresponding metadata
    [H_model] = \
        retrieve_map(H_model_file_name)
    
    # figure out the file path of the PNG picture of the histogram to be plotted
    H_model_plot_file_path = os.getcwd() + '/' + H_model_plot_file_name + '.png'

    # figure out the spacing of the tick labels for both axes
    x_tick_labels = np.arange(x_axis_start, x_axis_stop + x_tick_spacing, x_tick_spacing)
    y_tick_labels = np.arange(y_axis_start, y_axis_stop + y_tick_spacing, y_tick_spacing)

    # figure out the number of tick labels for both axes
    num_x_tick_labels = len(x_tick_labels)
    num_y_tick_labels = len(y_tick_labels)

    # figure out the actual ticks for both labels
    x_ticks = x_tick_labels/x_axis_step + 0.5*np.ones(num_x_tick_labels)
    y_ticks = y_tick_labels/y_axis_step + 0.5*np.ones(num_y_tick_labels)

    # flip the y axis ticks
    y_ticks = np.flip(y_ticks, 0)

    # create a figure
    fig = plt.figure()

    # plot the 2D histogram of the model patients
    ax = sns.heatmap(H_model, cbar_kws={'label':'probability of sampling patient from a point'})
    plt.xlabel('monthly seizure count mean')
    plt.ylabel('monthly seizure count standard deviation')
    plt.title(H_model_plot_title)

    # make the x-axis labels make sense
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation='horizontal')

    # make the y-axis labels make sense
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, rotation='horizontal')

    # save the histogram as a PNG file in the same folder as this script
    fig.savefig( H_model_plot_file_path )

    # close the figure
    plt.close(fig)
'''

'''
def plot_histograms(H_model_1_file_name, H_model_1_metadata_file_name, H_model_1_plot_file_name, H_model_1_plot_title,
                    H_model_2_file_name, H_model_2_metadata_file_name, H_model_2_plot_file_name, H_model_2_plot_title,
                    x_tick_spacing, y_tick_spacing):


    Purpose:

        This function plots the histograms of NV models 1 and 2.

    Inputs:

        1) H_model_1_file_name:
                
            (string) - the name of the file containing the histogram to be plotted for NV model 1

        2) H_model_1_metadata_file_name:
                
            (string) - the name of the file containing the metadata for the histogram to be plotted for NV model 1

        3) H_model_1_plot_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the 

                       histogram of the NV model 1 patients

        4) H_model_1_plot_title:
        
            (string) - the title of the histogram of NV model 1 to be plotted

        5) H_model_2_file_name:
                        
            (string) - the name of the file containing the histogram to be plotted for NV model 1

        6) H_model_2_metadata_file_name:
                        
            (string) - the name of the file containing the metadata for the histogram to be plotted for NV model 2

        7) H_model_2_plot_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the 

                       histogram of the NV model 2 patients

        8) H_model_2_plot_title:
                
            (string) - the title of the histogram of NV model 2 to be plotted

        9) x_tick_spacing:
                
            (float) - the spacing in between each of the labelled x-axis ticks

        10) y_tick_spacing:
        
            (float) - the spacing in between each of the labelled y-axis ticks

    Outputs:

        Technically None

    # plot the histogram of all the NV model 1 patients
    plot_histogram(H_model_1_file_name, 
                   H_model_1_metadata_file_name, H_model_1_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_1_plot_title)

    # plot the histogram of all the NV model 2 patients
    plot_histogram(H_model_2_file_name, 
                   H_model_2_metadata_file_name, H_model_2_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_2_plot_title)
'''


def plot_set_of_maps(RR50_file_name, RR50_plot_file_name, RR50_plot_with_power_curves_file_name, RR50_plot_title, RR50_plot_with_power_curves_title,
                     MPC_file_name,  MPC_plot_file_name,  MPC_plot_with_power_curves_file_name,  MPC_plot_title,  MPC_plot_with_power_curves_title,
                     TTP_file_name,  TTP_plot_file_name,  TTP_plot_with_power_curves_file_name,  TTP_plot_title,  TTP_plot_with_power_curves_title,
                     directory, min_req_base_sz_count, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step,
                     x_tick_spacing, y_tick_spacing, min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round):
    '''

    Purpose:

        This function plots a set of maps: the 50% responder data map, the 50% responder data map with power law curves, the median percent change map,
    
        the median percent change data map with power law curves, the time-to-prerandomization data map, and the time-to-prerandomization data map with 

        power law curves. The phrase 'data map' refers to one of four types of data: expected placebo arm response, expected drug arm response, statistical

        power, and type-1 error.

    Inputs:

        1) RR50_file_name:
        
            (string) - the name of the file containing the 50% responder rate data map to be plotted

        2) RR50_metadata_file_name:
        
            (string) - the name of the file containing the metadata for the 50% responder rate data map to be plotted

        3) RR50_plot_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the 50% responder rate data map

        4) RR50_plot_with_power_curves_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the 50% responder rate data map with power law curves

        5) RR50_plot_title:
        
            (string) - the title of the 50% responder rate data map to be plotted
        
        6) RR50_plot_with_power_curves_title:
        
            (string) - the title of the 50% responder rate data map with power law curves to be plotted
        
        7) MPC_file_name: 
                
            (string) - the name of the file containing the median percent change data map to be plotted

        8) MPC_metadata_file_name:
                
            (string) - the name of the file containing the metadata for the median percent change data map to be plotted

        9) MPC_plot_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the median percent change data map

        11) MPC_plot_with_power_curves_file_name:
                
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the median percent change data map with power law curves

        12) MPC_plot_title:
                
            (string) - the title of the median percent change data map to be plotted
        
        13) MPC_plot_with_power_curves_title:
        
            (string) - the title of the median percent change data map with power law curves to be plotted
        
        14) TTP_file_name:
                        
            (string) - the name of the file containing the time-to-prerandomization data map to be plotted

        15) TTP_metadata_file_name:
                        
            (string) - the name of the file containing the metadata for the time-to-prerandomization data map to be plotted

        16) TTP_plot_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the time-to-prerandomization data map

        17) TTP_plot_with_power_curves_file_name:
                        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the time-to-prerandomization data map with power law curves

        18) TTP_plot_title:
                        
            (string) - the title of the time-to-prerandomization data map to be plotted
        
        19) TTP_plot_with_power_curves_title:
        
            (string) - the title of the time-to-prerandomization data map with power law curves to be plotted
        
        20) x_tick_spacing:
        
            (float) - the spacing in between each of the labelled x-axis ticks on each map

        21) y_tick_spacing:
        
            (float) - the spacing in between each of the labelled y-axis ticks on each map
        
        22) min_power_law_slope:
        
            (float) - the minimum power law slope to be plotted
        
        23) max_power_law_slope:
        
            (float) - the maximum power law slope to be plotted
        
        24) power_law_slope_spacing:
        
            (float) - the size of the spaces in between each y-axis value

        25) legend_decimal_round:

            (int) - the decimal place to which the power law slopes should be rounded to
                    
                    within the legends of power law slopes
   
    Outputs:

        Technically None

    '''

    # plot the 50% responder rate data map
    plot_map(RR50_file_name, RR50_plot_file_name,
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             RR50_plot_title, legend_decimal_round)

    # plot the 50% responder rate data map with power law curves
    plot_map(RR50_file_name, RR50_plot_with_power_curves_file_name,
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             RR50_plot_with_power_curves_title, legend_decimal_round)
    
    # plot the median percent change data map
    plot_map(MPC_file_name, MPC_plot_file_name,
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             MPC_plot_title, legend_decimal_round)

    # plot the median percent change data map with power law curves
    plot_map(MPC_file_name, MPC_plot_with_power_curves_file_name,
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             MPC_plot_with_power_curves_title, legend_decimal_round)
    
    # plot the time-to-prerandomization data map
    plot_map(TTP_file_name, TTP_plot_file_name, 
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             TTP_plot_title, legend_decimal_round)

    # plot the time-to-prerandomization data map with power law curves
    plot_map(TTP_file_name, TTP_plot_with_power_curves_file_name, 
             directory, min_req_base_sz_count,
             x_axis_start, x_axis_stop, x_axis_step,
             y_axis_start, y_axis_stop, y_axis_step,
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             TTP_plot_with_power_curves_title, legend_decimal_round)


def main(directory, x_tick_spacing, y_tick_spacing, 
         min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round):
    '''

    Purpose:

        This function is the main function is the main function which coordinates all of the other 
    
        functions in this script. It generates the endpoint statistic maps as well as the histograms of 
    
        model 1 and model 2.

    Inputs:

        1) x_tick_spacing:
        
            (float) - the spacing in between each of the labelled x-axis ticks
        
        2) y_tick_spacing:

            (float) - the spacing in between each of the labelled y-axis ticks
        
        3) min_power_law_slope:

            (float) - the minimum power law slope to be plotted on the expected placebo reponse maps
        
        4) max_power_law_slope:

            (float) - the maximum power law slope to be plotted on the expected placebo reponse maps
        
        5) power_law_slope_spacing:

            (float) - the size of the spaces in between each y-axis value of the power law curves on the 
                      
                      expected placebo reponse maps
        
        6) legend_decimal_round:

            (int) - the decimal place to which the power law slopes should be rounded to
                    
                    within the legends of power law slopes
    
    Outputs:

        Technically None

    '''

    # set the names of the files associated/to-be-associated with the 50% responder rate expected placebo response map, median percent change expected placebo 
    # response map, and time-to-prerandomization expected placebo response map
    expected_placebo_RR50_file_name                        = 'expected_placebo_RR50_map'
    expected_placebo_RR50_plot_file_name                   = 'expected_placebo_RR50_plot'
    expected_placebo_RR50_plot_with_power_curves_file_name = 'expected_placebo_RR50_plot_with_power_curves'
    expected_placebo_MPC_file_name                         = 'expected_placebo_MPC_map'
    expected_placebo_MPC_plot_file_name                    = 'expected_placebo_MPC_plot'
    expected_placebo_MPC_plot_with_power_curves_file_name  = 'expected_placebo_MPC_plot_with_power_curves'
    expected_placebo_TTP_file_name                         = 'expected_placebo_TTP_map'
    expected_placebo_TTP_plot_file_name                    = 'expected_placebo_TTP_plot'
    expected_placebo_TTP_plot_with_power_curves_file_name  = 'expected_placebo_TTP_with_power_curves'

    # set the names of the files associated/to-be-associated with the 50% responder rate expected drug response map, median percent change expected drug 
    # response map, and time-to-prerandomization expected drug response map
    expected_drug_RR50_file_name                        = 'expected_drug_RR50_map'
    expected_drug_RR50_plot_file_name                   = 'expected_drug_RR50_plot'
    expected_drug_RR50_plot_with_power_curves_file_name = 'expected_drug_RR50_plot_with_power_curves'
    expected_drug_MPC_file_name                         = 'expected_drug_MPC_map'
    expected_drug_MPC_plot_file_name                    = 'expected_drug_MPC_plot'
    expected_drug_MPC_plot_with_power_curves_file_name  = 'expected_drug_MPC_plot_with_power_curves'
    expected_drug_TTP_file_name                         = 'expected_drug_TTP_map'
    expected_drug_TTP_plot_file_name                    = 'expected_drug_TTP_plot'
    expected_drug_TTP_plot_with_power_curves_file_name  = 'expected_drug_TTP_with_power_curves'

    # set the names of the files associated/to-be-associated with the 50% responder rate statistical power map, median percent change statistical power map, 
    # and time-to-prerandomization statistical power map
    RR50_stat_power_file_name                        = 'RR50_stat_power_map'
    RR50_stat_power_plot_file_name                   = 'RR50_stat_power_plot'
    RR50_stat_power_plot_with_power_curves_file_name = 'RR50_stat_power_plot_with_power_curves'
    MPC_stat_power_file_name                         = 'MPC_stat_power_map'
    MPC_stat_power_plot_file_name                    = 'MPC_stat_power_plot'
    MPC_stat_power_plot_with_power_curves_file_name  = 'MPC_stat_power_plot_with_power_curves'
    TTP_stat_power_file_name                         = 'TTP_stat_power_map'
    TTP_stat_power_plot_file_name                    = 'TTP_stat_power_plot'
    TTP_stat_power_plot_with_power_curves_file_name  = 'TTP_stat_power_plot_with_power_curves'

    # set the names of the files associated/to-be-associated with the 50% responder rate type-1 error map, median percent change type-1 error map, 
    # and time-to-prerandomization type-1 error map
    RR50_type_1_error_file_name                        = 'RR50_type_1_error_map'
    RR50_type_1_error_plot_file_name                   = 'RR50_type_1_error_plot'
    RR50_type_1_error_plot_with_power_curves_file_name = 'RR50_type_1_error_plot_with_power_curves'
    MPC_type_1_error_file_name                         = 'MPC_type_1_error_map'
    MPC_type_1_error_plot_file_name                    = 'MPC_type_1_error_plot'
    MPC_type_1_error_plot_with_power_curves_file_name  = 'MPC_type_1_error_plot_with_power_curves'
    TTP_type_1_error_file_name                         = 'TTP_type_1_error_map'
    TTP_type_1_error_plot_file_name                    = 'TTP_type_1_error_plot'
    TTP_type_1_error_plot_with_power_curves_file_name  = 'TTP_type_1_error_plot_with_power_curves'

    # set the titles of the plots to be generated for the 50% responder rate placebo response maps
    expected_placebo_RR50_plot_title                   = "Expected placebo arm 50% responder rate"
    expected_placebo_RR50_plot_with_power_curves_title = "Expected placebo arm 50% responder rate with power law curves"
    expected_placebo_MPC_plot_title                    = "Expected placebo arm median percent change"
    expected_placebo_MPC_plot_with_power_curves_title  = "Expected placebo arm median percent change with power law curves"
    expected_placebo_TTP_plot_title                    = "Expected placebo arm time-to-prerandomization"
    expected_placebo_TTP_plot_with_power_curves_title  = "Expected placebo arm time-to-prerandomization with power law curves"

    # set the titles of the plots to be generated for the 50% responder rate placebo response maps
    expected_drug_RR50_plot_title                   = "Expected drug arm 50% responder rate"
    expected_drug_RR50_plot_with_power_curves_title = "Expected drug arm 50% responder rate with power law curves"
    expected_drug_MPC_plot_title                    = "Expected drug arm median percent change"
    expected_drug_MPC_plot_with_power_curves_title  = "Expected drug arm median percent change with power law curves"
    expected_drug_TTP_plot_title                    = "Expected drug arm time-to-prerandomization"
    expected_drug_TTP_plot_with_power_curves_title  = "Expected drug arm time-to-prerandomization with power law curves"

    # set the titles of the plots to be generated for the statistical power maps
    RR50_stat_power_plot_title                   = "50% responder rate statistical power"
    RR50_stat_power_plot_with_power_curves_title = "50% responder rate statistical power with power law curves"
    MPC_stat_power_plot_title                    = "Median percent change statistical power"
    MPC_stat_power_plot_with_power_curves_title  = "Median percent change statistical power with power law curves"
    TTP_stat_power_plot_title                    = "Time-to-prerandomization statistical power"
    TTP_stat_power_plot_with_power_curves_title  = "Time-to-prerandomization statistical power with power law curves"

    # set the titles of the plots to be generated for the type-1 error maps
    RR50_type_1_error_plot_title                   = "50% responder rate type-1 error"
    RR50_type_1_error_plot_with_power_curves_title = "50% responder rate type-1 error with power law curves"
    MPC_type_1_error_plot_title                    = "Median percent change type-1 error"
    MPC_type_1_error_plot_with_power_curves_title  = "Median percent change type-1 error with power law curves"
    TTP_type_1_error_plot_title                    = "Time-to-prerandomization type-1 error"
    TTP_type_1_error_plot_with_power_curves_title  = "Time-to-prerandomization type-1 error with power law curves"

    '''
    # set the names of the files associated with the NV model 1 histogram
    H_model_1_file_name          = 'H_model_1_hist' 
    H_model_1_metadata_file_name = 'H_model_1_hist_metadata'
    H_model_1_plot_file_name     = 'H_model_1_plot'

    # set the names of the files associated with the NV model 2 histogram
    H_model_2_file_name          = 'H_model_2_hist' 
    H_model_2_metadata_file_name = 'H_model_2_hist_metadata'
    H_model_2_plot_file_name     = 'H_model_2_plot'

    # set the titles of the histograms for both Model 1 and Model 2
    H_model_1_plot_title = "Model 1 patient population"
    H_model_2_plot_title = "Model 2 patient population"
    '''

    # set the file name of the meta-data text file, which is hardcoded into the software, along with its absolute file path
    meta_data_file_name = 'meta_data.txt'
    meta_data_file_path = directory + '/' + meta_data_file_name

    # read the relevant information from the meta-data text file
    with open( meta_data_file_path, 'r' ) as meta_data_text_file:

        # read information about the monthly seizure count mean axis
        x_axis_start = int( meta_data_text_file.readline() )
        x_axis_stop = int( meta_data_text_file.readline() )
        x_axis_step = int( meta_data_text_file.readline() )

        # read information about the monthly seizure count standard deviation axis
        y_axis_start = int( meta_data_text_file.readline() )
        y_axis_stop = int( meta_data_text_file.readline() )
        y_axis_step = int( meta_data_text_file.readline() )

        # read information about the maximum of the minimum required number of seizures in the baseline period
        max_min_req_base_sz_count = int( meta_data_text_file.readline() )

    for min_req_base_sz_count in range(max_min_req_base_sz_count + 1):

        # plot all of the endpoint statistic maps having to do with the expected placebo response
        plot_set_of_maps(expected_placebo_RR50_file_name,  expected_placebo_RR50_plot_file_name, expected_placebo_RR50_plot_with_power_curves_file_name, 
                         expected_placebo_RR50_plot_title, expected_placebo_RR50_plot_with_power_curves_title,
                         expected_placebo_MPC_file_name,   expected_placebo_MPC_plot_file_name,  expected_placebo_MPC_plot_with_power_curves_file_name,  
                         expected_placebo_MPC_plot_title,  expected_placebo_MPC_plot_with_power_curves_title,
                         expected_placebo_TTP_file_name,   expected_placebo_TTP_plot_file_name,  expected_placebo_TTP_plot_with_power_curves_file_name,  
                         expected_placebo_TTP_plot_title,  expected_placebo_TTP_plot_with_power_curves_title,
                         directory, min_req_base_sz_count, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step,
                         x_tick_spacing, y_tick_spacing, min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round)
    
        # plot all of the endpoint statistic maps having to do with the expected drug response
        plot_set_of_maps(expected_drug_RR50_file_name,  expected_drug_RR50_plot_file_name, expected_drug_RR50_plot_with_power_curves_file_name, 
                         expected_drug_RR50_plot_title, expected_drug_RR50_plot_with_power_curves_title,
                         expected_drug_MPC_file_name,   expected_drug_MPC_plot_file_name, expected_drug_MPC_plot_with_power_curves_file_name,  
                         expected_drug_MPC_plot_title,  expected_drug_MPC_plot_with_power_curves_title,
                         expected_drug_TTP_file_name,   expected_drug_TTP_plot_file_name, expected_drug_TTP_plot_with_power_curves_file_name,
                         expected_drug_TTP_plot_title,  expected_drug_TTP_plot_with_power_curves_title,
                         directory, min_req_base_sz_count, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step,
                         x_tick_spacing, y_tick_spacing, min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round)
    
        # plot all of the endpoint statistic maps having to do with the statistical power
        plot_set_of_maps(RR50_stat_power_file_name,  RR50_stat_power_plot_file_name, RR50_stat_power_plot_with_power_curves_file_name, 
                         RR50_stat_power_plot_title, RR50_stat_power_plot_with_power_curves_title,
                         MPC_stat_power_file_name,   MPC_stat_power_plot_file_name, MPC_stat_power_plot_with_power_curves_file_name,
                         MPC_stat_power_plot_title,  MPC_stat_power_plot_with_power_curves_title,
                         TTP_stat_power_file_name,   TTP_stat_power_plot_file_name, TTP_stat_power_plot_with_power_curves_file_name, 
                         TTP_stat_power_plot_title,  TTP_stat_power_plot_with_power_curves_title,
                         directory, min_req_base_sz_count, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step,
                         x_tick_spacing, y_tick_spacing, min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round)
    
        # plot all of the endpoint statistic maps having to do with the type_1 error
        plot_set_of_maps(RR50_type_1_error_file_name,  RR50_type_1_error_plot_file_name, RR50_type_1_error_plot_with_power_curves_file_name, 
                         RR50_type_1_error_plot_title, RR50_type_1_error_plot_with_power_curves_title,
                         MPC_type_1_error_file_name,   MPC_type_1_error_plot_file_name, MPC_type_1_error_plot_with_power_curves_file_name,  
                         MPC_type_1_error_plot_title,  MPC_type_1_error_plot_with_power_curves_title,
                         TTP_type_1_error_file_name,   TTP_type_1_error_plot_file_name, TTP_type_1_error_plot_with_power_curves_file_name,  
                         TTP_type_1_error_plot_title,  TTP_type_1_error_plot_with_power_curves_title,
                         directory, min_req_base_sz_count, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step,
                         x_tick_spacing, y_tick_spacing, min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round)
    
    '''
    # plot the histograms of NV models 1 and 2
    plot_histograms(H_model_1_file_name, H_model_1_metadata_file_name, H_model_1_plot_file_name, H_model_1_plot_title,
                    H_model_2_file_name, H_model_2_metadata_file_name, H_model_2_plot_file_name, H_model_2_plot_title,
                    x_tick_spacing, y_tick_spacing)
    '''


if(__name__=='__main__'):

    # take in the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('array', nargs='+')
    args = parser.parse_args()
    arg_array = args.array

    # take in the directory containing all the JSON files from the user
    directory = arg_array[0]

    # obtain the spacing in between each of the labelled x-axis and y-axis ticks for all maps and histograms
    x_tick_spacing = float(arg_array[2])
    y_tick_spacing = float(arg_array[3])

    # obtain the slopes of the power law curves to be plotted
    min_power_law_slope = float(arg_array[4])
    max_power_law_slope = float(arg_array[5])
    power_law_slope_spacing = float(arg_array[6])
    legend_decimal_round = int(arg_array[7])

    # run the main function
    main(directory, x_tick_spacing, y_tick_spacing, 
         min_power_law_slope, max_power_law_slope, power_law_slope_spacing, legend_decimal_round)
