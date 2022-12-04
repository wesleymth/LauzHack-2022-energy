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

tracker = EmissionsTracker(output_dir ='logs', output_file="log.csv", project_name="LinearRegressor", log_level='error')
"""

py"tracker".start()
machine(LinearRegressor(), X, Y)|>fit!
py"tracker".stop()

### Testing Ridge Regressor

py"""
tracker = EmissionsTracker(output_dir ='logs', project_name="RidgeRegressor", log_level='error')
"""

py"tracker".start()
machine(RidgeRegressor(), X, Y)|>fit!
py"tracker".stop()

### Testing Ridge Regressor

py"""
tracker = EmissionsTracker(output_dir ='logs', project_name="LassoRegressor", log_level='error')
"""

py"tracker".start()
machine(LassoRegressor(), X, Y)|>fit!
py"IP".stop()


#### Test NN networks

