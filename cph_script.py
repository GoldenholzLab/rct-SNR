import numpy as np
import pandas as pd
import subprocess
import os


def estimate_hazard_ratio(placebo_TTP_times,
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
    treatment_arms         = 1 - np.int_(treatment_arms_str == "C")

    data = np.array([TTP_times, events, treatment_arms, treatment_arms_str]).transpose()
    pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms', 'treatment_arms_str']).to_csv(relative_tmp_file_path)
    process = subprocess.Popen(['Rscript', 'calculate_cph_power.R', relative_tmp_file_path, str(alpha), str(num_placebo_patients), str(num_drug_patients)], stdout=subprocess.PIPE)
    values = process.communicate()[0].decode().split()
    os.remove(relative_tmp_file_path)

    RR    = float(values[1])
    pE    = float(values[3])
    pC    = float(values[5])
    power = float(values[7])

    return [RR, pE, pC, power]


def calculate_hazard_function_from_unsorted_TTP_times(num_testing_days,
                                                      num_patients,
                                                      TTP_times):

    event_density = np.zeros(num_testing_days)
    for testing_day_index in range(num_testing_days):
        event_density[testing_day_index] = np.sum( TTP_times == testing_day_index + 1 )
    event_distribution = np.cumsum(event_density)
    survivor_curve = num_patients - event_distribution
    hazard_function = event_density/survivor_curve

    return hazard_function


num_placebo_patients = 4
num_drug_patients    = 4
alpha                = 0.05
num_testing_days     = 84
tmp_file_name        = 'tmp'
placebo_TTP_times    = np.array([  21,   15,   45,    85])
placebo_events       = np.array([True, True, True, False])
drug_TTP_times       = np.array([  24,   56,   78,    23])
drug_events          = np.array([True, True, True,  True])

placebo_hazard_function = \
    calculate_hazard_function_from_unsorted_TTP_times(num_testing_days,
                                                      num_placebo_patients,
                                                      placebo_TTP_times)

drug_hazard_function = \
    calculate_hazard_function_from_unsorted_TTP_times(num_testing_days,
                                                      num_drug_patients,
                                                      drug_TTP_times)

print(placebo_hazard_function)
print(drug_hazard_function)

'''
[RR, pE, pC, power] = \
    estimate_hazard_ratio(placebo_TTP_times,
                          placebo_events,
                          drug_TTP_times,
                          drug_events,
                          alpha,
                          num_placebo_patients,
                          num_drug_patients,
                          tmp_file_name)

print([RR, pE, pC, power])
'''
