'''
This file contains two methods that relates to data pre-processing.
Both of the methods deal with datasets that are processed before, i.e., reject outliers, remove N/A values, etc.
Datasets used here: traindata00.csv, traindata11.csv, traindata22.csv, and traindata33.csv.
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

'''
encode(dataframe):
1. encode features that are string type
2. each feature value is replaced by an int code
'''
def encode(dataframe):

    dataframe = dataframe.replace('Education', 0)
    dataframe = dataframe.replace('Office', 1)
    dataframe = dataframe.replace('Banking/financial services', 2)
    dataframe = dataframe.replace('Entertainment/public assembly', 3)
    dataframe = dataframe.replace('Food sales and service', 4)
    dataframe = dataframe.replace('Healthcare', 5)
    dataframe = dataframe.replace('Lodging/residential', 6)
    dataframe = dataframe.replace('Manufacturing/industrial', 7)
    dataframe = dataframe.replace('Mixed use', 8)
    dataframe = dataframe.replace('Parking', 9)
    dataframe = dataframe.replace('Public services', 10)
    dataframe = dataframe.replace('Religious worship', 11)
    dataframe = dataframe.replace('Retail', 12)
    dataframe = dataframe.replace('Technology/science', 13)
    dataframe = dataframe.replace('Services', 14)
    dataframe = dataframe.replace('Utility', 15)
    dataframe = dataframe.replace('Warehouse/storage', 16)
    dataframe = dataframe.replace('Other', 17)

    return dataframe

'''
getTrainData():
For reading dataset for each type of energy:
1. read in processed training data
2. encode features that are string type
3. delete column that is useless for modeling (e.g. id)
4. return the edited dataset
'''
def getTrainDate():

    Train0 = pd.read_csv('traindata00.csv', sep=',')
    Train0 = encode(Train0)
    Train0 = Train0.to_numpy()
    Train0 = np.delete(Train0, 0, axis=1)

    Train1 = pd.read_csv('traindata11.csv', sep=',')
    Train1 = encode(Train1)
    Train1 = Train1.to_numpy()
    Train1 = np.delete(Train1, 0, axis=1)

    Train2 = pd.read_csv('traindata22.csv', sep=',')
    Train2 = encode(Train2)
    Train2 = Train2.to_numpy()
    Train2 = np.delete(Train2, 0, axis=1)

    Train3 = pd.read_csv('traindata33.csv', sep=',')
    Train3 = encode(Train3)
    Train3 = Train3.to_numpy()
    Train3 = np.delete(Train3, 0, axis=1)

    return Train0, Train1, Train2, Train3

'''
splitData(TrainData):
1. split data into target and features
2. delete feature columns that are useless for modeling (e.g. id)
3. normalize feature values
'''
def splitData(TrainData):
    target = TrainData[:, 3]
    features = np.delete(TrainData, [0, 1, 2, 3], axis=1)
    return target, features
