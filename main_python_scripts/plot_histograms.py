import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string
import textwrap
from PIL import Image
import io
import matplotlib.font_manager as fm


if(__name__=='__main__'):

    model_errors_file_name =     sys.argv[1]
    num_bins               = int(sys.argv[2])

    endpoint_names = ['RR50', 'MPC', 'TTP']

    reg_prop  = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Regular.ttf')
    bold_prop = fm.FontProperties(fname='/Users/juanromero/Documents/GitHub/rct-SNR/Calibri Bold.ttf')

    #fig, ax = plt.subplots(1, 3, figsize=(10,6))
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
        current_ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        for tick in current_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        current_ax.set_xlabel('percent error',                 fontproperties=reg_prop, fontsize=14)
        current_ax.set_ylabel('fraction of prediction errors', fontproperties=reg_prop, fontsize=14)
        current_ax.set_title( formatted_title,                 fontproperties=reg_prop, fontsize=16)
        current_ax.set_xlim([-40, 40])
        current_ax.set_ylim([0, 0.12])
        current_ax.text(-0.3, 1, string.ascii_uppercase[endpoint_name_index] + ')', 
                        fontproperties=bold_prop, transform=current_ax.transAxes, size=25)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = .55)

    png1 = io.BytesIO()
    fig.savefig(png1, dpi = 300, bbox_inches = 'tight', format='png')
    png2 = Image.open(png1)
    png2.save('Romero-fig2.tiff')
    png1.close()

    #plt.savefig('statistical power prediction error histograms')
    #plt.show()
