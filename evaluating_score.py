# import library
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import os

p=os.getcwd() + r'\tlvmc-parkinsons-freezing-gait-prediction/'
sample_sub = pd.read_csv(p+'our_test_submission.csv')
sub = pd.read_csv(p+'submission.csv')


merged_sub= sample_sub.merge(sub,'inner',on='Id')




start_hesistation_event=average_precision_score(np.array(merged_sub["StartHesitation_x"]), np.array(merged_sub["StartHesitation_y"]))
turn_event=average_precision_score(np.array(merged_sub["Turn_x"]), np.array(merged_sub["Turn_y"]))
walking_event=average_precision_score(np.array(merged_sub["Walking_x"]), np.array(merged_sub["Walking_y"]))
print(start_hesistation_event, turn_event, walking_event)
print((start_hesistation_event + turn_event + walking_event) / 3)