import numpy as np
import json
import sys
import matplotlib.pyplot as plt


if(__name__=='__main__'):

    model_errors_file_name =     sys.argv[1]
    num_bins               = int(sys.argv[2])
    endpoint_name          =     sys.argv[3]

    with open(endpoint_name + '_' + model_errors_file_name + '.json', 'r') as json_file:
        model_errors = np.array(json.load(json_file))

    model_percent_errors = 100*model_errors

    plt.figure()
    plt.hist(model_percent_errors, bins=num_bins, density=True)
    plt.xlabel('percent error')
    plt.title('histogram of ' + endpoint_name + ' statistical power prediction error')
    plt.savefig(endpoint_name + ' statistical power prediction error histogram')
