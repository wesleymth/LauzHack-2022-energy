import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump
from data_processing import rename_categorical_cols


def train_models(dataset = 'energy_dataset.csv') :
    #Data preprocessing
    if type(dataset) == str : dataset= pd.read_csv(dataset)
    dataset_X = dataset.iloc[:,:-1]
    costs = dataset.iloc[:,-1]

    one_hot_df = one_hot_encoding(dataset_X)

    #Linear Regression Method with Ridge Cross Validation
    ridge = RidgeCV().fit(one_hot_df, costs)

    #XGBoost
    xgboost = GradientBoostingRegressor(random_state=0).fit(one_hot_df, costs)
    dump(ridge, 'ridge.joblib')
    dump(xgboost, 'xgboost.joblib')
    return ridge, xgboost

def one_hot_encoding(dataset) :
    dataset = rename_categorical_cols(dataset)
    one_hot_encoder = make_column_transformer((OneHotEncoder(drop ='first'), dataset.columns.str.contains('name')), remainder='passthrough')
    one_hot_df = pd.DataFrame(one_hot_encoder.fit_transform(dataset), columns=one_hot_encoder.get_feature_names_out())
    return one_hot_df