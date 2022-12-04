using Pkg
Pkg.activate(joinpath(@__DIR__, "Project.toml"))
using PyCall, DataFrames, MLJ, MLJLinearModels, Random, Conda
cd(@__DIR__)

Random.seed!(42)

X = DataFrame(randn((1000000,100)), :auto)
Y = randn(1000000)
results = DataFrame("CPU_count"=> String[], "CPU_vendor_id"=>String[], "CPU_GHz"=>Int[],"core_architecture"=>String[],
"memory_available_B"=>Int[], "swap_free_B"=>Int[], "OS"=>String[], "model_name"=>String[], "nb_sample"=>Int[], "nb_preds"=>Int[])
### Testing Linear Regressor

py"""
import re
import platform
import psutil
import cpuinfo

import GPUtil

def get_cpu_features():
    
    cpu_infos = cpuinfo.get_cpu_info()
    return {"CPU_count": cpu_infos['count'],
            "CPU_vendor_id": cpu_infos['vendor_id_raw'],
            "CPU_GHz": float(re.findall("\d+\.\d+", cpu_infos['hz_advertised_friendly'])[0]),
            "core_architecture" : cpu_infos['arch_string_raw']
            }

def get_system_features():
    return {
        "os" : platform.system()
    }
        

def get_GPU_features():

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
"""

py"""
import cpuinfo
print(cpuinfo.get_cpu_info())
"""
py"""
print(a)
"""
py"""

def train_data_generator(D_sup_N:bool=False, dataset_sizes:list=[20,500,1000]):
   
    if D_sup_N :
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes]
    else : 
        Xs_train = [np.random.randn(nb_sample, nb_pred) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
        ys_train = [np.random.randint(0,2,nb_sample) for nb_sample in dataset_sizes for nb_pred in dataset_sizes if nb_pred <= nb_sample]
    
    return Xs_train, ys_train
"""

py"""
from codecarbon import EmissionsTracker
from datetime import datetime
import numpy as np
"""

py"cpuinfo.get_cpu_info".()

# py"""
# train_data = train_data_generator()
# """

py"""
tracker = EmissionsTracker(output_dir ='logs', output_file="log.csv", project_name="LinearRegressor", log_level='error')
tracker.flush()
"""

datax, datay = py"train_data_generator".()
datax = [DataFrame(d, :auto) for d in datax]
for (i,d) in pairs(datax)
    py"tracker".start()
    machine(LinearRegressor(), d, datay[i])|>fit!
    py"tracker".stop()
end

### Testing Ridge Regressor

py"""
tracker = EmissionsTracker(output_dir ='logs', project_name="RidgeRegressor", log_level='error')
"""

datax, datay = py"train_data_generator".()
datax = [DataFrame(d, :auto) for d in datax]
for (i,d) in pairs(datax)
    py"tracker".start()
    machine(RidgeRegressor(), d, datay[i])|>fit!
    py"tracker".stop()
end

### Testing Ridge Regressor

py"""
tracker = EmissionsTracker(output_dir ='logs', project_name="LassoRegressor", log_level='error')
"""

datax, datay = py"train_data_generator".()
datax = [DataFrame(d, :auto) for d in datax]
for (i,d) in pairs(datax)
    py"tracker".start()
    machine(LassoRegressor(), d, datay[i])|>fit!
    py"tracker".stop()
end


#### Test NN networks

