from codecarbon import EmissionsTracker
import numpy as np
import sklearn as skl
from inspect import getmembers
import time
from tqdm import tqdm
#from dataset_initiator import dataset_initiator
from energy_extractor_intel import *
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
import GPUtil
import psutil
import platform
import re
import cpuinfo
import pandas as pd

def get_cpu_features():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    cpu_infos = cpuinfo.get_cpu_info()
    return {"CPU_count": cpu_infos['count'],
            "CPU_vendor_id": cpu_infos['vendor_id_raw'],
            "CPU_GHz": float(re.findall("\d+\.\d+", cpu_infos['hz_advertised_friendly'])[0]),
            "core_architecture" : cpu_infos['arch_string_raw']
            }

def get_system_features():
    # TODO : Add more relevant features.
    return {
        "os" : platform.system()
    }
        

def get_GPU_features():
    """If GPU is available, returns relevant GPU features used to create a energy consumption DataFrame"""
    # TODO: Add more relevant features
    if len(GPUtil.getGPUs()) != 0:
        return {
            "GPU_name" : GPUtil.getGPUs()[0].name
        }

def get_memory_features():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "memory_available_B" : mem.available,
        "swap_free_B" : swap.free
    }
    
def extract_energy(path_to_log_info:str)->list:
    df = pd.read_csv(os.path.join(path_to_log_info)).dropna()
    return df['Cumulative Processor Energy_0(mWh)'].iloc[-1]

np.random.seed()
def dataset_generator(nb_dataset_=5, all_sklearn=False) : 
    Xs_train, ys_train = train_data_generator(D_sup_N = False, nb_dataset = nb_dataset_)
    #dataset = dataset_initiator()
    
    dall = {}
    for d in [get_cpu_features(), get_memory_features(), get_system_features()]:
        dall.update(d)
    
    
    dataset = pd.DataFrame([],columns=list(dall.keys()) + ["model", "nb_samples", "nb_preds", "Cumul_Proc_Energy(mWh)"])
    non_working_models = []
    
    if all_sklearn :
        models = getmembers(skl.svm) + getmembers(skl.linear_model) + getmembers(skl.tree) + getmembers(skl.neural_network) + getmembers(skl.preprocessing) + getmembers(skl.ensemble)
    else : 
        #models = getmembers(linear_model)
        models = [LinearRegression, Ridge]
    i = 0
    
                        
    for f in tqdm(models):
        try:
            for X, y in zip(Xs_train, ys_train):
                #fit model and record energy consumption
                IPG = IntelPowerGadget(duration=3,
                                    resolution=100,
                                    #output_dir ='logs',
                                    log_file_name='log_file.csv')
                time.sleep(5)   
                IPG._setup_cli()
                     
                #f[1]().fit(X,y)
                f().fit(X,y)
                time.sleep(5)
                IPG._log_values()
                
                
                
                dataset.loc[i] = list(dall.values()) + [f.__name__, X.shape[0], X.shape[1], extract_energy('log_file.csv')]
                
                
                #rest for the processor to avoid successiv computation that could biased the energy consumption measurements
                
                i += 1
        
        except:
            #extract the model for which the train didn't work (issues with the parameter in general)
            non_working_models.append(f)
            pass
    
    return dataset

def train_data_generator(D_sup_N=False, nb_dataset=5) :
    #dataset_sizes = np.around(np.logspace(1,nb_dataset,nb_dataset)).astype(int)
    dataset_sizes = [100,2000,5000]
    if D_sup_N :
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
    else : 
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
    
    return Xs_train, ys_train


