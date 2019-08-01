import numpy as np
import pandas as pd
import subprocess


TTP_times      = np.array([21,   15,   45,    84,   24,   56,    78,    23])
events         = np.array([True, True, False, True, True, False, False, True])
treatment_arms = np.array([0,    0,    0,     0,    1,    1,     1,     1])

data = np.array([TTP_times, events, treatment_arms]).transpose()
pd.DataFrame(data, columns=['TTP_times', 'events', 'treatment_arms']).to_csv('tmp.csv')
process = subprocess.Popen(['Rscript', 'estimate_hazard_ratio.R', 'tmp.csv'], stdout=subprocess.PIPE)
postulated_log_hazard_ratio = float(process.communicate()[0].decode().split()[1])
print(postulated_log_hazard_ratio)