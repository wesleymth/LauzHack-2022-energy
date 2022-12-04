from codecarbon import EmissionsTracker
import numpy as np
import sklearn as skl
from inspect import getmembers
import time
from tqdm import tqdm
#from dataset_initiator import dataset_initiator
from energy_extractor_intel import *
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import linear_model
from hardware_features_extractor import *
import pandas as pd
import random

from datetime import datetime


np.random.seed(42)
random.seed(42)

def dataset_generator(dataset_sizes = [100,2000,5000], all_sklearn=False) : 
    Xs_train, ys_train = train_data_generator(D_sup_N = False, dataset_sizes = dataset_sizes)
    #dataset = dataset_initiator()
    
    dall = {}
    for d in [get_cpu_features(), get_memory_features(), get_system_features()]:
        dall.update(d)
    
    
    dataset = pd.DataFrame([],columns=list(dall.keys()) + ["model_name", "nb_samples", "nb_preds"])
    non_working_models = []
    
    if all_sklearn :
        models = getmembers(skl.svm) + getmembers(skl.linear_model) + getmembers(skl.tree) + getmembers(skl.neural_network) + getmembers(skl.preprocessing) + getmembers(skl.ensemble)
    else : 
        models = getmembers(linear_model)
        #models = [LinearRegression, Ridge]
        
        #IPG = IntelPowerGadget(duration=2, resolution=1000, #output_dir ='logs', log_file_name='log_file.csv')
    tracker = EmissionsTracker(output_file="sub_dataset_energy"+datetime.now().strftime("%H-%M-%S")+".csv", log_level='error')   
    for f in tqdm(models):
        try:
            for X, y in zip(Xs_train, ys_train):
                #fit model and record energy consumption 
                           
                tracker.start()
                f[1]().fit(X,y)
                #f().fit(X,y)
                tracker.stop()
                #IPG._log_values()
                
                dataset.loc[len(dataset)] = list(dall.values()) + [f.__name__, X.shape[0], X.shape[1]]
                
                
        except:
            #extract the model for which the train didn't work (issues with the parameter in general)
            non_working_models.append(f)
            pass
    dataset.to_csv("Model_features"+datetime.now().strftime("%H-%M-%S")+".csv")
    return dataset

def train_data_generator(D_sup_N=False, dataset_sizes = [100,2000,5000]) :
    #dataset_sizes = np.around(np.logspace(1,nb_dataset,nb_dataset)).astype(int)
    if D_sup_N :
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
    else : 
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
    
    return Xs_train, ys_train


