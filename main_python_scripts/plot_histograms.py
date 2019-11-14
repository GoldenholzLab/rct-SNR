import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
import io


if(__name__=='__main__'):

    model_errors_file_name =     sys.argv[1]
    num_bins               = int(sys.argv[2])

    endpoint_names = ['RR50', 'MPC', 'TTP']
    fig, ax = plt.subplots(1, 3, figsize=(20,6))

    for endpoint_name_index in range(len(endpoint_names)):

        endpoint_name = endpoint_names[endpoint_name_index]
        current_ax = ax[endpoint_name_index]

        with open(endpoint_name + '_' + model_errors_file_name + '.json', 'r') as json_file:

            model_errors = np.array(json.load(json_file))

        model_percent_errors = 100*model_errors

        long_title = 'histogram of ' + endpoint_name + ' statistical power prediction error'
        formatted_title = '\n'.join(textwrap.wrap(long_title, 30))

        current_ax.hist(model_percent_errors, bins=num_bins, density=True)
        current_ax.set_xlabel('percent error')
        current_ax.title.set_text(formatted_title)
        current_ax.set_xlim([-60, 60])
        current_ax.set_ylim([0, 0.45])
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = .15)

    png1 = io.BytesIO()
    fig.savefig(png1, dpi = 600, bbox_inches = 'tight', format='png')
    png2 = Image.open(png1)
    png2.save('Romero-fig2.tiff')
    png1.close()

    #plt.savefig('statistical power prediction error histograms')
    #plt.show()
