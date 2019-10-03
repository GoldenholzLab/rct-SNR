from keras import Input
from keras.layers import Conv2D
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Dense
import keras.models as models
import sys


def build_model(num_monthly_means,
                num_monthly_std_devs):

    placebo_arm_hist_input_tensor = Input(shape=(num_monthly_std_devs, num_monthly_means, 1))
    placebo_arm_2D_conv_tensor    = Conv2D(32, (5,5), activation='relu')(placebo_arm_hist_input_tensor)
    placebo_arm_2D_conv_tensor    = MaxPooling2D((3,3))(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = Conv2D(64, (3,3), activation='relu')(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = MaxPooling2D((2,2))(placebo_arm_2D_conv_tensor)
    placebo_arm_flattened_tensor  = Flatten()(placebo_arm_2D_conv_tensor)
    placebo_arm_hidden_layer      = Dense(32, activation='relu')(placebo_arm_flattened_tensor)
    placebo_arm_hidden_layer      = Dense(16, activation='relu')(placebo_arm_hidden_layer)
    
    drug_arm_hist_input_tensor = Input(shape=(num_monthly_std_devs, num_monthly_means, 1))
    drug_arm_2D_conv_tensor    = Conv2D(32, (5,5), activation='relu')(drug_arm_hist_input_tensor)
    drug_arm_2D_conv_tensor    = MaxPooling2D((3,3))(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = Conv2D(64, (3,3), activation='relu')(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = MaxPooling2D((2,2))(drug_arm_2D_conv_tensor)
    drug_arm_flattened_tensor  = Flatten()(drug_arm_2D_conv_tensor)
    drug_arm_hidden_layer      = Dense(32, activation='relu')(drug_arm_flattened_tensor)
    drug_arm_hidden_layer      = Dense(16, activation='relu')(drug_arm_hidden_layer)

    concatenated_placebo_and_drug_tensor = concatenate([placebo_arm_hidden_layer, drug_arm_hidden_layer])
    pseudo_power_output_tensor = Dense(32, activation='relu')(concatenated_placebo_and_drug_tensor)
    pseudo_power_output_tensor = Dense(16, activation='relu')(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(8, activation='relu')(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(4, activation='relu')(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(1, activation='sigmoid')(pseudo_power_output_tensor)

    RR50_stat_power_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], puseduo_power_output_tensor)
    RR50_stat_power_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    return RR50_stat_power_model


def take_inputs_from_commmand_shell():

    monthly_mean_min    = int(sys.argv[1])
    monthly_mean_max    = int(sys.argv[2])
    monthly_std_dev_min = int(sys.argv[3])
    monthly_std_dev_max = int(sys.argv[4])

    RR50_stat_power_model_file_name = sys.argv[5]

    return [monthly_mean_min,    monthly_mean_max,
            monthly_std_dev_min, monthly_std_dev_max,
            RR50_stat_power_model_file_name]


if(__name__=='__main__'):

    [monthly_mean_min,    monthly_mean_max,
     monthly_std_dev_min, monthly_std_dev_max,
     RR50_stat_power_model_file_name          ] = \
         take_inputs_from_commmand_shell()

    num_monthly_means    = monthly_mean_max    - (monthly_mean_min    - 1)
    num_monthly_std_devs = monthly_std_dev_max - (monthly_std_dev_min - 1)

    RR50_stat_power_model = \
        build_model(num_monthly_means,
                    num_monthly_std_devs)
    
    RR50_stat_power_model.save(RR50_stat_power_model_file_name)
