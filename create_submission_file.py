import numpy as np
import pandas as pd
import os
from tqdm import tqdm

OUR_MAIN_DIR=os.getcwd() + r'\tlvmc-parkinsons-freezing-gait-prediction/'


dataset= 'tdcsfog'
OUR_TEST_DATA= OUR_MAIN_DIR +"ourtest/"
TRAIN_DATA= OUR_MAIN_DIR +"test/" + dataset

df_res = pd.DataFrame()


for root, dirs, files in os.walk(OUR_TEST_DATA):
    for name in tqdm(files):
        f = os.path.join(root, name)
        query_datatype = pd.read_csv(f)
        query_datatype["file"] = name.replace(".csv", "")
        query_datatype["Id"] = query_datatype["file"] + "_"+ query_datatype["Time"].apply(str)
        df_res = pd.concat([df_res, query_datatype])
df_res = df_res.drop(["AccV","AccML","AccAP","Time"], axis = 1)

df_res.to_csv(OUR_MAIN_DIR+'our_test_submission.csv')
