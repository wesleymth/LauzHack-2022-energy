using Pkg
Pkg.activate(joinpath(@__DIR__, "Project.toml"))
using PyCall, DataFrames, MLJ, MLJLinearModels, Random, Conda, CSV
cd(@__DIR__)

Random.seed!(42)

X = DataFrame(randn((1000000,100)), :auto)
Y = randn(1000000)
results = DataFrame("CPU_count"=> String[], "CPU_vendor_id"=>String[], "CPU_GHz"=>Int[],"core_architecture"=>String[],
"memory_available_B"=>Int[], "swap_free_B"=>Int[], "OS"=>String[], "model_name"=>String[], "nb_sample"=>Int[], "nb_preds"=>Int[])
### Testing Linear Regressor

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

py"""
tracker = EmissionsTracker(output_dir ='logs', output_file="log.csv", project_name="LinearRegressor", log_level='error')
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

# table = CSV.read("../data/Model_features08-12-18_nico.csv",DataFrame)
# t2 = CSV.read("logs/log.csv",DataFrame)
# t3 = CSV.read("logs/emissions.csv", DataFrame)
# table = table[1:18,]