import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def retrieve_map(data_map_file_name, data_map_metadata_file_name):
    '''

    This function retrives a specified data map from the file system along with

    the metadata needed to help plot that data map, both of which are assumed to

    be stored in JSON files located in the same folder as this python script.

    Inputs:

        1) data_map_file_name:

            (string) - the name of the JSON file containing the actual data map
        
        2) data_map_metadata_file_name:

            (string) - the name of the JSON file containing the metadata needed to 
                       
                       plot the actual data map
        
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

    '''

    # create the string representation of the file path for the data map
    data_map_file_path = os.getcwd() + '/' + data_map_file_name + '.json'

    # create the string representation of the file path for the data map metadata
    data_map_metadata_file_path = os.getcwd() + '/' + data_map_metadata_file_name + '.json'

    # open the data map file
    with open(data_map_file_path, 'r') as map_storage_file:

        # load the data map into memory
        data_map = np.array(json.load(map_storage_file))

    # open the data map metadata file
    with open(data_map_metadata_file_path, 'r') as map_metadata_storage_file:

        # load the data map metadata file into memory
        metadata = np.array(json.load(map_metadata_storage_file))
    
    # access all the pieces of metadata from the metadata file
    x_axis_start = metadata[0]
    x_axis_stop = metadata[1]
    x_axis_step = metadata[2]
    y_axis_start = metadata[3]
    y_axis_stop = metadata[4]
    y_axis_step = metadata[5]
    
    return [data_map, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step]


def plot_power_law_curves(ax, min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
                x_axis_start, x_axis_stop, x_axis_step, 
                y_axis_start, y_axis_stop, y_axis_step):
    '''

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

    return ax


def plot_map(data_map_file_name, 
             data_map_metadata_file_name, data_plot_file_name, 
             x_tick_spacing, y_tick_spacing, plot_curves,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             map_title):
    '''

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

    Outputs:

        Technically None

    '''

    # retrieve the map and the map metadata
    [data_map, x_axis_start, x_axis_stop, x_axis_step, y_axis_start, y_axis_stop, y_axis_step] = \
        retrieve_map(data_map_file_name, data_map_metadata_file_name)

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
                                   y_axis_start, y_axis_stop, y_axis_step)

    # save the figure as a PNG file in the same folder as this script
    fig.savefig( data_plot_file_path )


def plot_histogram(H_model_file_name, 
                   H_model_metadata_file_name, H_model_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_plot_title):
    '''

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

    '''

    # retrieve the histogram of the NV model and its corresponding metadata
    [H_model, 
     x_axis_start, x_axis_stop, x_axis_step, 
     y_axis_start, y_axis_stop, y_axis_step] = \
        retrieve_map(H_model_file_name, H_model_metadata_file_name)
    
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


def main(RR50_stat_power_file_name,  RR50_stat_power_metadata_file_name, RR50_stat_power_plot_file_name, RR50_stat_power_plot_with_power_curves_file_name,
         MPC_stat_power_file_name,   MPC_stat_power_metadata_file_name,  MPC_stat_power_plot_file_name,  MPC_stat_power_plot_with_power_curves_file_name,
         TTP_stat_power_file_name,   TTP_stat_power_metadata_file_name,  TTP_stat_power_plot_file_name,  TTP_stat_power_plot_with_power_curves_file_name,
         RR50_stat_power_plot_title, RR50_stat_power_plot_with_power_curves_title, 
         MPC_stat_power_plot_title,  MPC_stat_power_plot_with_power_curves_title,
         TTP_stat_power_plot_title,  TTP_stat_power_plot_with_power_curves_title,
         H_model_1_file_name, H_model_1_metadata_file_name, H_model_1_plot_file_name, H_model_1_plot_title,
         H_model_2_file_name, H_model_2_metadata_file_name, H_model_2_plot_file_name, H_model_2_plot_title,
         x_tick_spacing, y_tick_spacing, 
         min_power_law_slope, max_power_law_slope, power_law_slope_spacing):
    '''

    This function is the main function is the main function which coordinates all of the other functions in this script. It generates the statistical

    power maps as well as the histograms of model 1 and model 2.

    Inputs:

        1) RR50_stat_power_file_name:

            (string) - the name of the file containing the 50% responder rate statistical power map to 
                       
                       be plotted
        
        2) RR50_stat_power_metadata_file_name:

            (string) - the name of the file containing the metadata for the 50% responder rate statistical 
                       
                       power map to be plotted
        
        3) RR50_stat_power_plot_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the 50% responder rate statistical power map
        
        4) RR50_stat_power_plot_with_power_curves_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the 50% responder rate statistical power map with power law curves
        
        5) MPC_stat_power_file_name:

            (string) - the name of the file containing the median percent change statistical power 
            
                       map to be plotted
        
        6) MPC_stat_power_metadata_file_name:
        
            (string) - the name of the file containing the metadata for the median percent change 
                       
                       statistical power map to be plotted
        
        7) MPC_stat_power_plot_file_name: 
        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the  median percent change statistical power map
        
        8) MPC_stat_power_plot_with_power_curves_file_name:

            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the median percent change statistical power map with power law curves
        
        9) TTP_stat_power_file_name:
        
            (string) - the name of the file containing the expected time-to-prerandomization statistical 
                       
                       power map to be plotted

        10) TTP_stat_power_metadata_file_name:
                
            (string) - the name of the file containing the metadata for the time-to-prerandomization
                       
                       statistical power map to be plotted
        
        11) TTP_stat_power_plot_file_name:
                
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the time-to-prerandomization statistical power map
        
        12) TTP_stat_power_plot_with_power_curves_file_name:
        
            (string) - the name of the file that will eventually contain the PNG image of the plot 
                       
                       of the time-to-prerandomization statistical power map with power law curves
        
        13) RR50_stat_power_plot_title:

            (string) - the title of the RR50 statistical power map that will be in the corresponding 
            
                       PNG image
        
        14) RR50_stat_power_plot_with_power_curves_title:

            (string) - the title of the RR50 statistical power map with power law curves that will be 
            
                       in the corresponding PNG image
        
        15) MPC_stat_power_plot_title:
        
            (string) - the title of the MPC statistical power map that will be in the corresponding 
            
                       PNG image
        
        16) MPC_stat_power_plot_with_power_curves_title:

            (string) - the title of the MPC statistical power map with power law curves that will be 
            
                       in the corresponding PNG image
        
        17) TTP_stat_power_plot_title:
                
            (string) - the title of the TTP statistical power map that will be in the corresponding 
            
                       PNG image
        
        18) TTP_stat_power_plot_with_power_curves_title:

            (string) - the title of the TTP statistical power map with power law curves that will be 
            
                       in the corresponding PNG image
        
        19) H_model_1_file_name

            (string) - the name of the file containing the histogram of all the model 1 patients to be plotted

        20) H_model_1_metadata_file_name

            (string) - the name of the file containing the metadata for the histogram of all the model 1 patients 
                       
                       to be plotted

        21) H_model_1_plot_file_name

            (string) - the name of the file that will eventually contain the PNG image of the 

                       histogram of the NV model 1 patients

        22) H_model_1_plot_title
        
            (string) - the title of the histogram of all the NV model 1 patients to be plotted

        23) H_model_2_file_name

            (string) - the name of the file containing the histogram of of all the model 2 patients to be plotted

        24) H_model_2_metadata_file_name

            (string) - the name of the file containing the metadata for the histogram of all the model 2 patients 
                       
                       to be plotted

        25) H_model_2_plot_file_name

            (string) - the name of the file that will eventually contain the PNG image of the 

                       histogram of the NV model 2 patients

        26) H_model_2_plot_title

            (string) - the title of the histogram of all the NV model 2 patients to be plotted

        27) x_tick_spacing:
        
            (float) - the spacing in between each of the labelled x-axis ticks
        
        28) y_tick_spacing:

            (float) - the spacing in between each of the labelled y-axis ticks
        
        29) min_power_law_slope:

            (float) - the minimum power law slope to be plotted on the expected placebo reponse maps
        
        30) max_power_law_slope:

            (float) - the maximum power law slope to be plotted on the expected placebo reponse maps
        
        31) power_law_slope_spacing:

            (float) - the size of the spaces in between each y-axis value of the power law curves on the 
                      
                      expected placebo reponse maps
    
    Outputs:

        Technically None

    '''

    # plot the expected 50% responder rate placebo response map
    plot_map(RR50_stat_power_file_name, 
             RR50_stat_power_metadata_file_name, RR50_stat_power_plot_file_name, 
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             RR50_stat_power_plot_title)

    # plot the expected 50% responder rate placebo response map with power law curves
    plot_map(RR50_stat_power_file_name, 
             RR50_stat_power_metadata_file_name, RR50_stat_power_plot_with_power_curves_file_name, 
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             RR50_stat_power_plot_with_power_curves_title)
    
    # plot the expected median percent change placebo response map
    plot_map(MPC_stat_power_file_name, 
             MPC_stat_power_metadata_file_name, MPC_stat_power_plot_file_name, 
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             MPC_stat_power_plot_title)

    # plot the expected median percent change placebo response map with power law curves
    plot_map(MPC_stat_power_file_name,
             MPC_stat_power_metadata_file_name, MPC_stat_power_plot_with_power_curves_file_name, 
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             MPC_stat_power_plot_with_power_curves_title)
    
    # plot the expected time-to-prerandomization placebo response map
    plot_map(TTP_stat_power_file_name, 
             TTP_stat_power_metadata_file_name, TTP_stat_power_plot_file_name, 
             x_tick_spacing, y_tick_spacing, False,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             TTP_stat_power_plot_title)

    # plot the expected time-to-prerandomization placebo response map with power law curves
    plot_map(TTP_stat_power_file_name, 
             TTP_stat_power_metadata_file_name, TTP_stat_power_plot_with_power_curves_file_name, 
             x_tick_spacing, y_tick_spacing, True,
             min_power_law_slope, max_power_law_slope, power_law_slope_spacing,
             TTP_stat_power_plot_with_power_curves_title)
    
    # plot the histogram of all the NV model 1 patients
    plot_histogram(H_model_1_file_name, 
                   H_model_1_metadata_file_name, H_model_1_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_1_plot_title)

    # plot the histogram of all the NV model 2 patients
    plot_histogram(H_model_2_file_name, 
                   H_model_2_metadata_file_name, H_model_2_plot_file_name, 
                   x_tick_spacing, y_tick_spacing, H_model_2_plot_title)


if(__name__=='__main__'):

    # take in the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('array', nargs='+')
    args = parser.parse_args()
    arg_array = args.array

    # obtain the names of the files associated with the 50% responder rate placebo response map
    RR50_stat_power_file_name = arg_array[0]
    RR50_stat_power_metadata_file_name = arg_array[1]
    RR50_stat_power_plot_file_name = arg_array[2]
    RR50_stat_power_plot_with_power_curves_file_name = arg_array[3]

    # set the titles of the plots to be generated for the 50% responder rate placebo response maps
    RR50_stat_power_plot_title = "Expected RR50 placebo response"
    RR50_stat_power_plot_with_power_curves_title = "Expected RR50 placebo response with power law curves"

    # obtain the names of the files associated with the median percent change placebo response map
    MPC_stat_power_file_name = arg_array[4]
    MPC_stat_power_metadata_file_name = arg_array[5]
    MPC_stat_power_plot_file_name = arg_array[6]
    MPC_stat_power_plot_with_power_curves_file_name = arg_array[7]

    # set the titles of the plots to be generated for the median percent change placebo response maps
    MPC_stat_power_plot_title = "Expected MPC placebo response"
    MPC_stat_power_plot_with_power_curves_title = "Expected MPC placebo response with power law curves"

    # obtain the names of the files associated with the median percent change placebo response map
    TTP_stat_power_file_name = arg_array[8]
    TTP_stat_power_metadata_file_name = arg_array[9]
    TTP_stat_power_plot_file_name = arg_array[10]
    TTP_stat_power_plot_with_power_curves_file_name = arg_array[11]

    # set the titles of the plots to be generated for the median percent change placebo response maps
    TTP_stat_power_plot_title = "Expected TTP placebo response"
    TTP_stat_power_plot_with_power_curves_title = "Expected TTP placebo response with power law curves"

    # obtain the names of the files associated with the NV model 1 histogram
    H_model_1_file_name = arg_array[12]
    H_model_1_metadata_file_name = arg_array[13]
    H_model_1_plot_file_name = arg_array[14]

    # obtain the names of the files associated with the NV model 1 histogram
    H_model_2_file_name = arg_array[15]
    H_model_2_metadata_file_name = arg_array[16]
    H_model_2_plot_file_name = arg_array[17]

    # set the titles of the histograms for both Model 1 and Model 2
    H_model_1_plot_title = "Model 1 patient population"
    H_model_2_plot_title = "Model 2 patient population"

    # obtain the spacing in between each of the labelled x-axis and y-axis ticks for all maps and histograms
    x_tick_spacing = float(arg_array[18])
    y_tick_spacing = float(arg_array[19])

    # obtain the slopes of the power law curves to be plotted
    min_power_law_slope = float(arg_array[20])
    max_power_law_slope = float(arg_array[21])
    power_law_slope_spacing = float(arg_array[22])

    # run the main function
    main(RR50_stat_power_file_name,  RR50_stat_power_metadata_file_name, RR50_stat_power_plot_file_name, RR50_stat_power_plot_with_power_curves_file_name,
         MPC_stat_power_file_name,   MPC_stat_power_metadata_file_name,  MPC_stat_power_plot_file_name,  MPC_stat_power_plot_with_power_curves_file_name,
         TTP_stat_power_file_name,   TTP_stat_power_metadata_file_name,  TTP_stat_power_plot_file_name,  TTP_stat_power_plot_with_power_curves_file_name,
         RR50_stat_power_plot_title, RR50_stat_power_plot_with_power_curves_title, 
         MPC_stat_power_plot_title,  MPC_stat_power_plot_with_power_curves_title,
         TTP_stat_power_plot_title,  TTP_stat_power_plot_with_power_curves_title,
         H_model_1_file_name, H_model_1_metadata_file_name, H_model_1_plot_file_name, H_model_1_plot_title,
         H_model_2_file_name, H_model_2_metadata_file_name, H_model_2_plot_file_name, H_model_2_plot_title,
         x_tick_spacing, y_tick_spacing, 
         min_power_law_slope, max_power_law_slope, power_law_slope_spacing)
