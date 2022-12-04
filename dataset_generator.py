from codecarbon import EmissionsTracker
import numpy as np
import sklearn as skl
from inspect import getmembers
import time
from tqdm import tqdm
#from dataset_initiator import dataset_initiator
from energy_extractor_intel import *

np.random.seed()
def dataset_generator(nb_dataset_=5, all_sklearn=False) : 
    Xs_train, ys_train = train_data_generator(D_sup_N = False, nb_dataset = nb_dataset_)
    #dataset = dataset_initiator()
    tracker = EmissionsTracker()
    non_working_models = []
    
    if all_sklearn :
        models = getmembers(skl.svm) + getmembers(skl.linear_model) + getmembers(skl.tree) + getmembers(skl.neural_network) + getmembers(skl.preprocessing) + getmembers(skl.ensemble)
    else : 
        models = getmembers(skl.linear_model)

    for f in tqdm(models):
        try:
            if(f[0][0]!='_'):
                for X,y in zip(Xs_train, ys_train):
                    #fit model and record energy consumption
                    IPG._setup_cli()
                    f[1]().fit(X,y)
                    IPG._log_values()
                    
                    #add sample to dataset
                    dataset.add_sample(f.__name__, X.shape[0], X.shape[1])
                    
                    #rest for the processor to avoid successiv computation that could biased the energy consumption measurements
                    time.sleep(2)
        except:
            #extract the model for which the train didn't work (issues with the parameter in general)
            non_working_models.append(f)
            pass
    dataset = 'finished !'
    return dataset

def train_data_generator(D_sup_N=True, nb_dataset=5) :
    dataset_sizes = np.around(np.logspace(1,nb_dataset,nb_dataset)).astype(int)
    if D_sup_N :
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample, nb_pred in zip(dataset_sizes, dataset_sizes)]
    else : 
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample, nb_pred in zip(dataset_sizes, dataset_sizes) if nb_pred <= nb_sample]
    ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes]
    return Xs_train, ys_train