from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, concatenate, Dense, Dropout, PReLU
import keras.models as models
from keras.utils import plot_model
import sys


def build_model(num_monthly_means,
                num_monthly_std_devs):

    placebo_arm_hist_input_tensor = Input(shape=(num_monthly_std_devs, num_monthly_means, 1))
    placebo_arm_2D_conv_tensor    = Conv2D(32, (2,2))(placebo_arm_hist_input_tensor)
    placebo_arm_2D_conv_tensor    = PReLU()(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = MaxPooling2D((2,2))(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = Conv2D(64, (2,2))(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = PReLU()(placebo_arm_2D_conv_tensor)
    placebo_arm_2D_conv_tensor    = MaxPooling2D((2,2))(placebo_arm_2D_conv_tensor)
    placebo_arm_flattened_tensor  = Flatten()(placebo_arm_2D_conv_tensor)
    placebo_arm_hidden_layer      = Dense(64)(placebo_arm_flattened_tensor)
    placebo_arm_hidden_layer      = PReLU()(placebo_arm_hidden_layer)
    placebo_arm_hidden_layer      = Dropout(0.3)(placebo_arm_hidden_layer)
    placebo_arm_hidden_layer      = Dense(32)(placebo_arm_hidden_layer)
    placebo_arm_hidden_layer      = PReLU()(placebo_arm_hidden_layer)
    placebo_arm_hidden_layer      = Dropout(0.3)(placebo_arm_hidden_layer)

    drug_arm_hist_input_tensor = Input(shape=(num_monthly_std_devs, num_monthly_means, 1))
    drug_arm_2D_conv_tensor    = Conv2D(32, (2,2))(drug_arm_hist_input_tensor)
    drug_arm_2D_conv_tensor    = PReLU()(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = MaxPooling2D((2,2))(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = Conv2D(64, (2,2))(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = PReLU()(drug_arm_2D_conv_tensor)
    drug_arm_2D_conv_tensor    = MaxPooling2D((2,2))(drug_arm_2D_conv_tensor)
    drug_arm_flattened_tensor  = Flatten()(drug_arm_2D_conv_tensor)
    drug_arm_hidden_layer      = Dense(64)(drug_arm_flattened_tensor)
    drug_arm_hidden_layer      = PReLU()(drug_arm_hidden_layer)
    drug_arm_hidden_layer      = Dropout(0.3)(drug_arm_hidden_layer)
    drug_arm_hidden_layer      = Dense(32)(drug_arm_hidden_layer)
    drug_arm_hidden_layer      = PReLU()(drug_arm_hidden_layer)
    drug_arm_hidden_layer      = Dropout(0.3)(drug_arm_hidden_layer)

    concatenated_placebo_and_drug_tensor = concatenate([placebo_arm_hidden_layer, drug_arm_hidden_layer])
    pseudo_power_output_tensor = Dense(32)(concatenated_placebo_and_drug_tensor)
    pseudo_power_output_tensor = PReLU()(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dropout(0.3)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(16)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = PReLU()(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dropout(0.3)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(8)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = PReLU()(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dropout(0.3)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(4)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = PReLU()(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dropout(0.3)(pseudo_power_output_tensor)
    pseudo_power_output_tensor = Dense(1, activation='sigmoid')(pseudo_power_output_tensor)

    stat_power_model = models.Model([placebo_arm_hist_input_tensor, drug_arm_hist_input_tensor], pseudo_power_output_tensor)
    stat_power_model.compile(optimizer='rmsprop', loss='mean_squared_error')

    return stat_power_model


def take_inputs_from_command_shell():
    
    monthly_mean_lower_bound    = int(sys.argv[1])
    monthly_mean_upper_bound    = int(sys.argv[2])
    monthly_std_dev_lower_bound = int(sys.argv[3])
    monthly_std_dev_upper_bound = int(sys.argv[4])

    generic_stat_power_model_file_name = sys.argv[5]
    endpoint_name                      = sys.argv[6]

    return [monthly_mean_lower_bound,    monthly_mean_upper_bound,
            monthly_std_dev_lower_bound, monthly_std_dev_upper_bound,
            generic_stat_power_model_file_name, endpoint_name]


if(__name__=='__main__'):

    [monthly_mean_lower_bound,    monthly_mean_upper_bound,
     monthly_std_dev_lower_bound, monthly_std_dev_upper_bound,
     generic_stat_power_model_file_name, endpoint_name] = \
         take_inputs_from_command_shell()

    num_monthly_means    = monthly_mean_upper_bound    - (monthly_mean_lower_bound    - 1)
    num_monthly_std_devs = monthly_std_dev_upper_bound - (monthly_std_dev_lower_bound - 1)
    stat_power_model_file_path = endpoint_name + '_' + generic_stat_power_model_file_name + '.h5'

    stat_power_model = \
        build_model(num_monthly_means,
                    num_monthly_std_devs)
    
    stat_power_model.save(stat_power_model_file_path)

    '''
    from PIL import Image
    plot_model(stat_power_model, to_file='stat_power_model.png', show_shapes=True)
    img = Image.open('stat_power_model.png')
    img.save('stat_power_model.tiff')
    '''
    