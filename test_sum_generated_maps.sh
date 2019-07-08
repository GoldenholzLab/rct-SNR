
directory='/Users/juanromero/Documents/Python_3_Files/test'
num_patients_per_model=10000
num_months_per_patient=24
shape_1=24.143
scale_1=297.366
alpha_1=284.024
beta_1=369.628
shape_2=111.313
scale_2=296.728
alpha_2=296.339
beta_2=243.719

inputs[0]=$directory
inputs[1]=$num_patients_per_model
inputs[2]=$num_months_per_patient
inputs[3]=$shape_1
inputs[4]=$scale_1
inputs[5]=$alpha_1
inputs[6]=$beta_1
inputs[7]=$shape_2
inputs[8]=$scale_2
inputs[9]=$alpha_2
inputs[10]=$beta_2

python sum_generated_maps.py ${inputs[@]}
