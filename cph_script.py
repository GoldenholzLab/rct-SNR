import numpy as np
import pandas as pd
import subprocess
import os


def generate_patient_pop_params(monthly_mean_min,
                                monthly_mean_max, 
                                monthly_std_dev_min, 
                                monthly_std_dev_max, 
                                num_theo_patients_per_trial_arm):

    patient_pop_monthly_param_sets = np.zeros((num_theo_patients_per_trial_arm , 2))

    for theo_patient_index in range(num_theo_patients_per_trial_arm ):

        overdispersed = False

        while(not overdispersed):

            monthly_mean    = np.random.randint(monthly_mean_min,    monthly_mean_max)
            monthly_std_dev = np.random.randint(monthly_std_dev_min, monthly_std_dev_max)

            if(monthly_std_dev > np.sqrt(monthly_mean)):

                overdispersed = True

        patient_pop_monthly_param_sets[theo_patient_index, 0] = monthly_mean
        patient_pop_monthly_param_sets[theo_patient_index, 1] = monthly_std_dev

    return patient_pop_monthly_param_sets


def estimate_analytical_power(placebo_TTP_times,
                   placebo_events,
                   drug_TTP_times,
                   drug_events,
                   alpha,
                   num_placebo_patients,
                   num_drug_patients,
                   tmp_file_name):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_TTP_times, drug_TTP_times)
    events                 = np.append(placebo_events, drug_events)
    treatment_arms_str     = np.append( np.array(num_placebo_patients*['C']) , np.array(num_drug_patients*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    process = subprocess.Popen(['Rscript', 'calculate_cph_power.R', relative_tmp_file_path, str(alpha), str(num_placebo_patients), str(num_drug_patients)], stdout=subprocess.PIPE)
    values = process.communicate()[0].decode().split()
    os.remove(relative_tmp_file_path)

    RR    = float(values[1])
    pC    = float(values[3])
    pE    = float(values[5])
    power = float(values[7])

    return [RR, pC, pE, power]


def estimate_analytical_power_2(placebo_TTP_times,
                     placebo_events,
                     drug_TTP_times,
                     drug_events,
                     num_placebo_patients,
                     num_drug_patients,
                     alpha):

    relative_tmp_file_path = tmp_file_name + '.csv'
    TTP_times              = np.append(placebo_TTP_times, drug_TTP_times)
    events                 = np.append(placebo_events, drug_events)
    treatment_arms_str     = np.append( np.array(num_placebo_patients*['C']) , np.array(num_drug_patients*['E']) )
    treatment_arms         = np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms']).to_csv(relative_tmp_file_path)
    process = subprocess.Popen(['Rscript', 'estimate_hazard_ratio.R', relative_tmp_file_path], stdout=subprocess.PIPE)
    postulated_hazard_ratio = float(process.communicate()[0].decode().split()[1])
    os.remove(relative_tmp_file_path)

    prob_fail_placebo = np.sum(placebo_events == True)/num_placebo_patients
    prob_fail_drug    = postulated_hazard_ratio*np.sum(drug_events    == True)/num_drug_patients   # this line is very suspicious

    command = ['Rscript', 'calculate_cph_power_2.R', str(num_drug_patients), str(num_placebo_patients), str(prob_fail_drug), str(prob_fail_placebo), str(postulated_hazard_ratio), str(alpha)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    power   = float(process.communicate()[0].decode().split()[1])
    
    return [postulated_hazard_ratio, 100*prob_fail_placebo, 100*prob_fail_drug, power]


num_placebo_patients = 6
num_drug_patients    = 6
alpha                = 0.05
num_testing_days     = 84
tmp_file_name        = 'tmp'
placebo_TTP_times    = np.array([  21,   15,   45,    67,    85,    4])
placebo_events       = np.array([True, True, True,  True, False, True])
drug_TTP_times       = np.array([  24,   56,   78,    85,    23,    8])
drug_events          = np.array([True, True, True, False,  True, True])


[RR, pC, pE, power] = \
    estimate_analytical_power(placebo_TTP_times,
                   placebo_events,
                   drug_TTP_times,
                   drug_events,
                   alpha,
                   num_placebo_patients,
                   num_drug_patients,
                   tmp_file_name)

[postulated_hazard_ratio, prob_fail_placebo, prob_fail_drug, power_2] = \
    estimate_analytical_power_2(placebo_TTP_times,
                     placebo_events,
                     drug_TTP_times,
                     drug_events,
                     num_placebo_patients,
                     num_drug_patients,
                     alpha)

print([postulated_hazard_ratio, np.round(prob_fail_placebo, 5), np.round(prob_fail_drug, 5), power_2])
print([RR, pC, pE, power])

