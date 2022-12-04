using Pkg
Pkg.activate(joinpath(@__DIR__, "Project.toml"))
using PyCall, DataFrames, MLJ, MLJLinearModels, Random, Conda


Random.seed!(0)

X = DataFrame(randn((1000000,100)), :auto)
Y = randn(1000000)

### Testing Linear Regressor

py"""
from codecarbon import EmissionsTracker
from datetime import datetime

tracker = EmissionsTracker(output_dir ='logs', output_file="sub_dataset_energy.csv", log_level='error')
"""

py"tracker".start()
machine(LinearRegressor(), X, Y)|>fit!
py"tracker".stop()

### Testing Ridge Regressor

py"""
IP = IntelPowerGadget(duration=2, resolution=1000,
log_file_name='RidgeRegressor.csv')

"""

py"IP"._setup_cli()
machine(RidgeRegressor(), X, Y)|>fit!
py"IP"._log_values()

### Testing Ridge Regressor

py"""
IP = IntelPowerGadget(duration=2, resolution=1000,
log_file_name='LassoRegressor.csv')
"""

py"IP"._setup_cli()
machine(LassoRegressor(), X, Y)|>fit!
py"IP"._log_values()


#### Test NN networks

