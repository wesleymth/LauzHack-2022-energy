import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
import sklearn.linear_model as skl
from inspect import getmembers
import time
from tqdm import tqdm
from datasets_generator import datasets_generator
from dataset_creator import create_dataset


Xs_train, ys_train = datasets_generator()
dataset = create_dataset()
tracker = EmissionsTracker()
non_working_models = []

for f in tqdm(getmembers(skl)):
    try:
        for X,y in zip(Xs_train, ys_train):
            #fit model and record energy consumption
            tracker.start()
            f[1]().fit(X,y)
            tracker.stop()
            
            #add sample to dataset
            dataset.add_sample(f.__name__, X.shape[0], X.shape[1])
            
            #rest for the processor to avoid successiv computation that could biased the energy consumption measurements
            time.sleep(2)
    except:
        #extract the model for which the train didn't work (issues with the parameter in general)
        non_working_models.append(f)
        pass