import pandas as pd
import numpy as np
import boto3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
import pickle
import sqlite3
from sqlalchemy import create_engine

#loads data from path
def load_data(): 
    s3_client = boto3.client('s3', region_name='us-east-2')
    bucket_name = 'harshil-storage'
    file_key = 'v_dem_indicators_2024.csv'
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    indicators = pd.read_csv(response.get("Body"))
    return indicators

#dropping uneeded columns and rows
def clean_data(indicators):
    indicators.drop(columns=['country_name', 'country_id', 'histname' ,'historical_date','project','historical',
                         'codingstart','codingend', 'codingstart_contemp', 'codingend_contemp',
                         'codingstart_hist', 'codingend_hist', 'gapstart1', 
                         'gapstart2', 'gapstart3', 'gapend1', 'gapend2', 'gapend3', 'gap_index',
                         'COWcode'], inplace=True)
    indicators = indicators[indicators["year"] >= 1980]
    indicators = indicators.sort_values(by=['country_text_id', 'year'])
    indicators_2023 = indicators[indicators["year"] == 2023]
    civil_war = indicators['e_civil_war']
    indicators.drop(columns=['e_civil_war'], inplace=True)
    indicators.insert(2, 'e_civil_war', civil_war)
    indicators = indicators.dropna(subset=['e_civil_war'])
    indicators = indicators.dropna(thresh = indicators.shape[0]*0.95, axis = 1)

    #dropping columns with codelow or codehigh in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='codelow')))]
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='codehigh')))]

    #dropping columns with sd in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='sd')))]

    #dropping columns with ord in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='ord')))]

    #dropping columns with mean in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='mean')))]

    #dropping columns with osp in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='osp')))]


    #dropping columns with nr in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='nr')))]

    #dropping columns with e_v2x in name
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='e_v2x')))]

    #removing columns that end with a number
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='\d+$')))]

    #removing columns that have the word region
    indicators = indicators[indicators.columns.drop(list(indicators.filter(regex='region')))]

    #setting index to country_text_id and year
    indicators.set_index(['country_text_id', 'year'], inplace=True)

    #dropping columns that are not type int or float
    indicators = indicators.select_dtypes(include=['int64', 'float64'])

    #separrating the data into features and target
    X = indicators.drop(columns=['e_civil_war'])
    y = indicators['e_civil_war']

    #dropping columns in the 2023 data that have NaN
    indicators_2023 = indicators_2023.dropna(thresh=indicators_2023.shape[0]*0.3, axis=1)

    #removing columns in X that are not in the 2023 data
    X = pd.DataFrame(X)
    X = X[X.columns.intersection(indicators_2023.columns)]

    #setting the index of indicators_2023 to country_text_id and year
    indicators_2023.set_index(['country_text_id', 'year'], inplace=True)

    #removing columns in indicators_2023 that are not in X
    indicators_2023 = indicators_2023[indicators_2023.columns.intersection(X.columns)]

    #saving 2023 data to sqlite database
    indicators_2023.reset_index(inplace=True)
    engine = create_engine('sqlite:///./data/2023_data.db')
    indicators_2023.to_sql('indicators_2023', engine, if_exists='replace', index=False)
    
    #return data
    return X, y


def impute_data(X):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X


def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Save scaler object
    with open('./models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return X

def apply_pca(X):
    pca = PCA(n_components=30)
    X = pca.fit_transform(X)
    # Save pca object
    with open('./models/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    return X

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

def undersample_data(X_train, y_train):
    rus = RandomOverSampler(random_state=69)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train

def preprocess_data(pca=True):
    indicators = load_data()
    X, y = clean_data(indicators)
    X = impute_data(X)
    X = scale_data(X)
    if pca:
        X = apply_pca(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, y_train = undersample_data(X_train, y_train)
    return X_train, X_test, y_train, y_test

def preprocess_data_predict(country, pca=True):
    conn = sqlite3.connect('./data/2023_data.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM indicators_2023 WHERE country_text_id = ?", (country,))
    data = cur.fetchall()
    columns = [description[0] for description in cur.description]
    data = np.array(data)
    data = data[:, 2:]
    data = data.astype(float) 
    if data.size == 0:
        return None
    #if 25 percent of the columns are NaN, return None
    if np.isnan(data).sum() > data.size*0.25:
        return None
    data = np.nan_to_num(data)
    # Load scaler and pca objects
    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    data = scaler.transform(data)
    if pca:
        with open('./models/pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        data = pca.transform(data)
    return data, columns
    

