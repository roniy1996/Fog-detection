import os
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import shutil


MAIN_DIR = ""
OUR_MAIN_DIR=""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ["AccV", "AccML", "AccAP"]
TARGETS = ["StartHesitation", "Turn", "Walking"]

N_EPOCHS = 1


# Reduce Memory Usage
# reference : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 @ARJANGROEN

def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage became: ", mem_usg, " MB")

    return df



def read_data(
        dataset,
        datatype,
        subject_id=None):
    metadata = pd.read_csv(MAIN_DIR + dataset + "_metadata.csv")

    DATA_ROOT = MAIN_DIR + datatype + "/" + dataset
    OUR_DATA_ROOT = "our" + datatype + "/" + dataset
    OUR_TRAIN_DATA = "ourtrain/" + dataset
    OUR_TEST_DATA = "ourtest/" + dataset


    df_res = pd.DataFrame()
    if datatype == "train":
        for root, dirs, files in os.walk(OUR_TRAIN_DATA):
            for name in tqdm(files):
                f = os.path.join(root, name)
                query_datatype = pd.read_csv(f)
                query_datatype["file"] = name.replace(".csv", "")
                df_res = pd.concat([df_res, query_datatype])
    else:
        for root, dirs, files in os.walk(OUR_TEST_DATA):
            for name in tqdm(files):
                f = os.path.join(root, name)
                query_datatype = pd.read_csv(f)
                query_datatype["file"] = name.replace(".csv", "")
                df_res = pd.concat([df_res, query_datatype])

    df_res = metadata.merge(df_res,
                            how='inner',
                            left_on='Id',
                            right_on='file')
    df_res = df_res.drop(["file"], axis=1)

    if datatype == "test":
        df_res = df_res.drop(["Walking", "Turn", "StartHesitation"], axis=1)

    df_res = reduce_memory_usage(df_res)

    return df_res


def read_data_(
        dataset="defog",
        datatype="train",
        subject_id=None):
    metadata = pd.read_csv(MAIN_DIR + dataset + "_metadata.csv")
    DATA_ROOT = datatype + "/" + dataset
    OUR_DATA_ROOT = "our" + datatype + "/" + dataset


    df_res = pd.DataFrame()
    for root, dirs, files in os.walk(OUR_DATA_ROOT):

        for name in tqdm(files):
            f = os.path.join(root, name)
            query_datatype = pd.read_csv(f)
            if dataset == "defog" and datatype == "train":
                # mask = query_datatype['Valid'].apply(str) == '0'
                new_query_datatype = query_datatype.loc[query_datatype['Valid'] == True]
                # new_query_datatype= query_datatype[~mask]
            else:
                new_query_datatype = query_datatype
            new_query_datatype["file"] = name.replace(".csv", "")
            df_res = pd.concat([df_res, new_query_datatype])

    if dataset == "tdcsfog":
        metadata = metadata.drop('Test', axis=1)
    df_res = metadata.merge(df_res,
                            how='inner',
                            left_on='Id',
                            right_on='file')

    df_res = df_res.drop(["file"], axis=1)

    if (dataset == "defog") and (datatype == "train"):
        df_res = df_res.drop('Valid', axis=1)
        df_res = df_res.drop('Task', axis=1)

    df_res = reduce_memory_usage(df_res)

    return df_res


#create combined train dataset

df_defog = read_data_(dataset="defog", datatype="train")
df_tdcsfog = read_data_(dataset="tdcsfog", datatype="train")
combined_df_train = pd.concat([df_defog, df_tdcsfog])

#create combined test dataset

df_defog_test = read_data(dataset="defog", datatype="test")
df_tdcsfog_test = read_data(dataset="tdcsfog", datatype="test")
combined_df_test = pd.concat([df_defog_test, df_tdcsfog_test])

class FOGDataset(Dataset):

    @staticmethod
    def encode_target(data, targets_list):
        conditions = []
        for target in targets_list:
            conditions.append((data[target] == 1))

        event = np.select(conditions, targets_list, default='Normal')
        le = LabelEncoder()
        return le.fit_transform(event)

    @staticmethod
    def get_features_target(data, features_list, datatype):
        if datatype == "train":
            features, target = data[features_list], data["target"]
            return features, target
        else:
            features = data[features_list]
            return features

    def __init__(self, dataset, datatype, features_list, targets_list, lookback):
        self.datatype = datatype
        # self.data = read_data(dataset = dataset, datatype = datatype)
        self.data = dataset
        self.features = features_list
        self.targets = targets_list
        self.data["Id_encoded"], _ = pd.factorize(self.data["Id"])
        self.lookback = lookback

        if datatype == "train":
            self.data = self.data[:10_000]
            self.data["target"] = FOGDataset.encode_target(self.data, self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.datatype == "train":
            features, targets = FOGDataset.get_features_target(self.data,
                                                               self.features,
                                                               self.datatype
                                                               )

            if idx < self.lookback:
                features = features[0: self.lookback]
                targets = targets[self.lookback]

            else:
                features = features[idx - self.lookback: idx]
                targets = targets[idx]

            features = torch.tensor(features.to_numpy(), dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            return features, targets
        else:
            features = FOGDataset.get_features_target(self.data,
                                                      self.features,
                                                      self.datatype
                                                      )

            if idx < self.lookback:
                features = features[0: self.lookback]

            else:
                features = features[idx - self.lookback: idx]

            features = torch.tensor(features.to_numpy(), dtype=torch.float32)

            return features


dataset_train = FOGDataset(
    dataset=combined_df_train,
    datatype="train",
    features_list=FEATURES,
    targets_list=TARGETS,
    lookback=2
)
dataset_test = FOGDataset(
    dataset=combined_df_test,
    datatype="test",
    features_list=FEATURES,
    targets_list=TARGETS,
    lookback=2
)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=1000, shuffle=False)

count = 0

for batch in dataloader_train:
    features, target = batch
    print("FEATURES EXAMPLES")
    print(features.shape)
    print(features)
    print("TARGET EXAMPLES")
    print(target)
    print("\n")
    if count > 1:
        break
    count += 1


    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            hidden_state = torch.zeros((self.num_layers, x.size(0), self.hidden_size), dtype=torch.float32)
            cell_state = torch.zeros((self.num_layers, x.size(0), self.hidden_size), dtype=torch.float32)

            out, _ = self.lstm(x, (hidden_state, cell_state))
            out = out[:, -1, :]
            out = self.fc1(out)
            return out



    def train(model, dataloader, loss_fn, optimizer):
        model.train()
        total_loss = 0
        for epoch in range(N_EPOCHS):
            mean_precision = []
            for (features, targets) in tqdm(dataloader):
                optimizer.zero_grad()
                preds = model(features)
                loss = loss_fn(preds, targets.long())
                mean_precision.append(loss.item())
                loss.backward()
                optimizer.step()

            print("Average Precision : ", np.mean(mean_precision))

        return model


    def predict(model, dataloader):
        model.eval()
        predictions = np.empty(len(dataset_test))
        count = 0
        for features in tqdm(dataloader):
            preds = model(features)
            preds = torch.argmax(preds, dim=1)
            preds = preds.numpy()
            predictions[count: count + len(preds)] = preds
            count += len(preds)

        return predictions


    INPUT_SIZE = len(FEATURES)
    HIDDEN_SIZE = 10
    NUM_LAYERS = 1
    NUM_CLASSES = 4
    PARAMS = {
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": NUM_CLASSES
    }
    model = LSTMNet(**PARAMS)

    loss_fn = CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = train(
        model,
        dataloader_train,
        loss_fn,
        optimizer
    )

    preds_combined = predict(model, dataloader_test)

    combined_df_test["y_pred"] = preds_combined

    sub_fmt = pd.read_csv("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv")

    sub = pd.DataFrame()
    for data in [combined_df_test]:
        temp = data.copy()
        temp["Id"] = temp.apply(lambda x: str(x.Id) + "_" + str(x.Time), axis=1)
        temp['StartHesitation'] = np.where(temp['y_pred'] == 1, 1, 0)
        temp['Turn'] = np.where(temp['y_pred'] == 2, 1, 0)
        temp['Walking'] = np.where(temp['y_pred'] == 3, 1, 0)
        temp = temp[["Id"] + TARGETS]
        sub = pd.concat([sub, temp])

        sub.to_csv("/kaggle/working/submission.csv", index=False)