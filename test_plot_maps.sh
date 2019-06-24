
inputs[0]='/Users/juanromero/Documents/GitHub/rct-SNR'

# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[1]=2
inputs[2]=2

# The slopes of the power law curves to be plotted
inputs[3]=0.5
inputs[4]=1.4
inputs[5]=0.1

# the decimal place to which all power law slopes should be rounded to on legends
inputs[6]=1

python plot_data.py ${inputs[@]}
