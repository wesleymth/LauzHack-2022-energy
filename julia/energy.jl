using Pkg
Pkg.activate(joinpath(@__DIR__, "Project.toml"))
using PyCall, DataFrames, MLJ, MLJLinearModels, Random

py"""
import shutil
import os
import subprocess
import sys
import time
class IntelPowerGadget:

    _osx_exec = "PowerLog"
    _osx_exec_backup = "/Applications/Intel Power Gadget/PowerLog"
    _windows_exec = "PowerLog3.0.exe"
    _windows_exec_backup = "C:\\Program Files\\Intel\\Power Gadget 3.6\\PowerLog3.0.exe"
    def __init__(
        self,
        output_dir: str = ".",
        duration=10,
        resolution=1000,
        log_file_name="powerlog.csv",
    ):
        self._log_file_path = os.path.join(output_dir, log_file_name)
        self._system = sys.platform.lower()
        self._duration = duration
        self._resolution = resolution
        self._setup_cli()

    def _setup_cli(self):

        if self._system.startswith("win"):
            if shutil.which(self._windows_exec):
                self._cli = shutil.which(
                    self._windows_exec
                )  # Windows exec is a relative path
            elif shutil.which(self._windows_exec_backup):
                self._cli = self._windows_exec_backup
            else:
                raise FileNotFoundError(
                    f"Intel Power Gadget executable not found on {self._system}"
                )
        elif self._system.startswith("darwin"):
            if shutil.which(self._osx_exec):
                self._cli = self._osx_exec
            elif shutil.which(self._osx_exec_backup):
                self._cli = self._osx_exec_backup
            else:
                raise FileNotFoundError(
                    f"Intel Power Gadget executable not found on {self._system}"
                )
        else:
            raise SystemError("Platform not supported by Intel Power Gadget")


    def _log_values(self):

        returncode = None
        if self._system.startswith("win"):
            print(subprocess.PIPE)
            returncode = subprocess.call(
                [
                    self._cli,
                    "-duration",
                    str(self._duration),
                    "-resolution",
                    str(self._resolution),
                    "-file",
                    self._log_file_path,
                ],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        elif self._system.startswith("darwin"):
            returncode = subprocess.call(
                f"'{self._cli}' -duration {self._duration} -resolution {self._resolution} -file {self._log_file_path} > /dev/null",  # noqa: E501
                shell=True,
            )
        else:
            return None
    

"""




Random.seed!(0)

X = DataFrame(randn((1000000,100)), :auto)
Y = randn(1000000)

### Testing Linear Regressor

py"""
IP = IntelPowerGadget(duration=2, resolution=1000,
log_file_name='LinearRegressor.csv')

"""

py"IP"._setup_cli()
machine(LinearRegressor(), X, Y)|>fit!
py"time.sleep"(15)
py"IP"._log_values()

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

