
directory='/Users/juanromero/Documents/Python_3_Files/test'
shape_1=24.143
scale_1=297.366
alpha_1=284.024
beta_1=369.628
shape_2=111.313
scale_2=296.728
alpha_2=296.339
beta_2=243.719

inputs[0]=$directory
inputs[1]=$shape_1
inputs[3]=$scale_1
inputs[4]=$alpha_1
inputs[5]=$beta_1
inputs[6]=$shape_2
inputs[7]=$scale_2
inputs[8]=$alpha_2
inputs[9]=$beta_2

python sum_generated_maps.py ${inputs[@]}
