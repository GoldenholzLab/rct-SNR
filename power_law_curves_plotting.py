import numpy as np
import matplotlib.pyplot as plt

mu_start = 0
mu_stop = 16
mu_step = 0.1
sigma_start = 0
sigma_stop = 16
power_law_slopes = np.array([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])

num_power_law_slopes = len(power_law_slopes)
mu_axis = np.arange(mu_start, mu_stop + mu_step, mu_step)

plt.figure()

for power_law_slope_index in range(num_power_law_slopes):

    power_law_slope = power_law_slopes[power_law_slope_index]
    sigma_axis = np.power(mu_axis, power_law_slope)

    plt.plot(mu_axis, sigma_axis)

plt.ylim([sigma_start, sigma_stop])
plt.legend(['slope: ' + str(power_law_slopes[power_law_slope_index]) for power_law_slope_index in range(num_power_law_slopes)])
plt.show()
