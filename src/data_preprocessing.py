import pandas as pd
import boto3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc

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

    return X, y


def impute_data(X):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    return X


def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def apply_pca(X):
    pca = PCA(n_components=30)
    X = pca.fit_transform(X)
    return X

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

def undersample_data(X_train, y_train):
    rus = RandomUnderSampler(random_state=69)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    return X_train, y_train

def preprocess_data():
    indicators = load_data()
    X, y = clean_data(indicators)
    X = impute_data(X)
    X = scale_data(X)
    X = apply_pca(X)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, y_train = undersample_data(X_train, y_train)
    return X_train, X_test, y_train, y_test


