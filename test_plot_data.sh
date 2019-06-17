
# The spacing in between each of the labelled x-axis and y-axis ticks
inputs[0]=2
inputs[1]=2

# The slopes of the power law curves to be plotted
inputs[2]=0.5
inputs[3]=1.4
inputs[4]=0.1

# the decimal place to which all power law slopes should be rounded to on legends
inputs[5]=1

python plot_data.py ${inputs[@]}
