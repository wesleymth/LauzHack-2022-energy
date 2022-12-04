import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump
from data_processing import rename_categorical_cols
import numpy as np


def train_models(dataset = 'energy_dataset.csv') :
    #Data preprocessing
    if type(dataset) == str : dataset= pd.read_csv(dataset)
    dataset_X = dataset.iloc[:,:-1]
    costs = dataset.iloc[:,-1]

    one_hot_df = one_hot_encoding(dataset_X)

    #Linear Regression Method with logarithmization of the costs
    linear = LinearRegression().fit(one_hot_df, np.log(costs))

    #XGBoost
    xgboost = GradientBoostingRegressor(random_state=0).fit(one_hot_df, costs)
    dump(linear, 'linear.joblib')
    dump(xgboost, 'xgboost.joblib')
    return linear, xgboost

"""def one_hot_encoding_test(dataset) :
    dataset = rename_categorical_cols(dataset)
    category_col = dataset.columns.str.contains('name')
    one_hot_encoder = make_column_transformer((OneHotEncoder(drop ='first'), category_col), remainder='passthrough')
    transformed = one_hot_encoder.fit_transform(dataset)
    one_hot_df = pd.DataFrame(transformed, columns=one_hot_encoder.get_feature_names_out())
    return one_hot_df, one_hot_encoder
"""
def one_hot_encoding(dataset) :
    #dataset = rename_categorical_cols(dataset)
    category_col = dataset.columns.str.contains('name')
    one_hot_encoder = make_column_transformer((OneHotEncoder(drop ='first'), category_col), remainder='passthrough')
    transformed = one_hot_encoder.fit_transform(dataset)
    one_hot_df = pd.DataFrame(transformed.todense(), columns=one_hot_encoder.get_feature_names_out())
    dump(one_hot_encoder, 'one_hot_encoder.joblib')
    return one_hot_df