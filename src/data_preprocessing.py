import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

#loads data from path
def load_data(file): 
    data = pd.read_csv(file)
    return data

#dropping uneeded columns and rows
def clean_data(indicators):
    indicators = indicators.drop(columns=['country_name', 'country_id', 'histname' ,'historical_date','project','historical',
                         'codingstart','codingend', 'codingstart_contemp', 'codingend_contemp',
                         'codingstart_hist', 'codingend_hist', 'gapstart1', 
                         'gapstart2', 'gapstart3', 'gapend1', 'gapend2', 'gapend3', 'gap_index',
                         'COWcode'], inplace=True)
    indicators = indicators[indicators.year >= 1980]
    indicators = indicators.sort_values(by=['country_text_id', 'year'])
    indicators_2023 = indicators[indicators.year == 2023]
    civil_war = indicators['e_civil_war']
    indicators.drop(columns=['e_civil_war'], inplace=True)
    indicators.insert(2, 'e_civil_war', civil_war)
    indicators = indicators.dropna(subset=['e_civil_war'])
    indicators = indicators.dropna(thresh = indicators.shape[0]*0.3, axis = 1)

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

    #getting list of columns with nr in name
    nr_columns = list(indicators.filter(regex='nr'))

    #getting subset of nr columns
    nr_indicators = indicators[nr_columns]

    #replacing values less than 3 with NaN
    nr_indicators.replace({3:None}, inplace=True)

    #dropping rows with NaN
    nr_indicators.dropna(axis=0, inplace=True)

    #only selecting rows that are in the nr_indicators
    indicators = indicators[indicators.index.isin(nr_indicators.index)]

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

    #dropping columns in the 2023 data that are more than 30% empty
    indicators_2023 = indicators_2023.dropna(thresh=indicators_2023.shape[0]*0.3, axis=1)

    #removing columns in X that are not in the 2023 data
    X = pd.DataFrame(X)
    X = X[X.columns.intersection(indicators_2023.columns)]

    return X, y

def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def apply_pca(X):
    pca = PCA(n_components=45)
    X = pca.fit_transform(X)
    return X

def undersample_data(X, y):
    rus = RandomUnderSampler(random_state=69)
    X, y = rus.fit_resample(X, y)
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

def preprocess_data(file):
    indicators = load_data(file)
    X, y = clean_data(indicators)
    X = scale_data(X)
    X = apply_pca(X)
    X, y = undersample_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test

