import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# heatmap parameters
x_axis_start = 0
x_axis_stop = 16
x_axis_step = 1

y_axis_start = 0
y_axis_stop = 16
y_axis_step = 1

x_tick_spacing = 1
y_tick_spacing = 1

# scatter plot parameters
num_points = 100
noise_level = 0.5
slope = (y_axis_stop - y_axis_start)/(x_axis_stop - x_axis_start)
intercept = 0

# power law parameters
max_power_law_slope = 1.3
min_power_law_slope = 0.5
power_law_slope_spacing = 0.1
legend_decimal_round = 1

x_array = np.arange(x_axis_start, x_axis_stop + x_axis_step, x_axis_step)
y_array = np.flip( np.arange(y_axis_start, y_axis_stop + y_axis_step, y_axis_step), 0 )

num_x = len(x_array)
num_y = len(y_array)

data_matrix = np.zeros((num_y, num_x))

index_matrix = np.zeros((num_y, num_x), dtype=tuple)

for y_index in range(num_y):

        for x_index in range(num_x):

                x = x_array[x_index]
                y = y_array[y_index]

                index_matrix[y_index, x_index] = (x, y)
                data_matrix[y_index, x_index] = x + y

x_values = np.random.uniform(x_axis_start + 1, x_axis_stop - 1, num_points)
noise = np.random.normal(0, noise_level, num_points)
y_values = slope*x_values + intercept + noise

power_law_slopes = np.arange(min_power_law_slope/power_law_slope_spacing, max_power_law_slope/power_law_slope_spacing + 1, 1)*power_law_slope_spacing

x_scale_ratio = num_x/(x_axis_stop - x_axis_start)
y_scale_ratio = num_y/(y_axis_stop - y_axis_start)

x_tick_labels = np.arange(x_axis_start, x_axis_stop + x_tick_spacing, x_tick_spacing)
y_tick_labels = np.arange(y_axis_start, y_axis_stop + y_tick_spacing, y_tick_spacing)
num_x_tick_labels = len(x_tick_labels)
num_y_tick_labels = len(y_tick_labels)

x_ticks = x_tick_labels/x_axis_step + 0.5*np.ones(num_x_tick_labels)
y_ticks = np.flip( y_tick_labels/y_axis_step + 0.5*np.ones(num_y_tick_labels), 0)

plt.figure()
ax = sns.heatmap(data_matrix)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, rotation='horizontal')
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels, rotation='horizontal')

plt.scatter(x_values*x_scale_ratio, (y_axis_stop - y_values)*y_scale_ratio )

for power_law_slope_index in range(len(power_law_slopes)):
        power_law_slope = power_law_slopes[power_law_slope_index]
        power_law_y_array = np.power(x_array, power_law_slope)
        plt.plot(x_array*x_scale_ratio, (y_axis_stop - power_law_y_array)*y_scale_ratio )
plt.legend( [ str(np.round(power_law_slopes[power_law_slope_index], legend_decimal_round)) for power_law_slope_index in range(len(power_law_slopes)) ] ) 

print(pd.DataFrame(data_matrix).to_string())

plt.show()
