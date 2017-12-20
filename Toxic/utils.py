import os
import pandas as pd
import seaborn as sns
from zipfile import ZipFile

_all_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

def unzip_all():
    lookupfor = ['sample_submission.csv.zip', 'test.csv.zip', 'train.csv.zip']
    for file in lookupfor:
        if file in os.listdir(os.path.join(".","data")):
            print("Unzip {}".format(file))
            ZipFile(os.path.join(".","data",file),'r').extract(file[:-4])
            os.rename(file[:-4], os.path.join("data",file[:-4]))
            print("Done")
        else:
            raise ValueError("{} not found in 'data'".format(file))
    
def load_dataframes(lookupfor = ['sample_submission.csv', 'test.csv', 'train.csv']):
    if any(e not in os.listdir('data') for e in lookupfor):
        print("Csv files not found, try unzipping...")
        unzip_all()
        print("Unzipping sucessfull")

    return [pd.read_csv(os.path.join('data',file)) for file in lookupfor]
    
def get_train_labels():
    train_df = load_dataframes(lookupfor = ['train.csv'])[0]
    return train_df[_all_labels]

