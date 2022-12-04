from dataset_generator import *
from joblib import dump, load
from data_processing import rename_categorical_cols
from train_model import one_hot_encoding

def predict(df):
    model = load('xgboost.joblib')
    return model.predict(df)

def energy(model, X):
    dall = {}
    for d in [get_cpu_features(), get_system_features()]:
        dall.update(d)
    dataset = pd.DataFrame([],columns=list(dall.keys()) + ["model_name", "nb_samples", "nb_preds"])
    
    dataset.loc[len(dataset)] = list(dall.values()) + [model.__name__, X.shape[0], X.shape[1]]
    #predictors = one_hot_encoding(dataset)
    #predict(predictors)
    onehot = load('one_hot_encoder.joblib')
    transformed = onehot.fit_transform(dataset)
    one_hot_df = pd.DataFrame(transformed, columns=onehot.get_feature_names_out())
    print(transformed)
    return predict(one_hot_df)
