import numpy as np
from sklearn.linear_model import LinearRegression
import shutil
import os
import subprocess
import sys

class IntelPowerGadget:
    """
        Set up IntePowerGadget software and associated path.
        Create a command line using this path to run the software.
        Create a csv file with all the results inside.
        Parameters can be modified like : duration / resolution / log_file_name
    """
    _osx_exec = "PowerLog"
    _osx_exec_backup = "/Applications/Intel Power Gadget/PowerLog"
    _windows_exec = "PowerLog3.0.exe"
    _windows_exec_backup = "C:\\Program Files\\Intel\\Power Gadget 3.6\\PowerLog3.0.exe"
    def __init__(
        self,
        output_dir: str = ".",
        duration=10,
        resolution=1000,
        log_file_name=".\\Data\\CPU_log_infos.csv",
    ):
        self._log_file_path = os.path.join(output_dir, log_file_name)
        self._system = sys.platform.lower()
        self._duration = duration
        self._resolution = resolution
        self._setup_cli()

    def _setup_cli(self):
        """
        Setup cli command to run Intel Power Gadget using the path
        """
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
        """
        Logs output from Intel Power Gadget command line to a file using PowerLog3.0
        Put the results in Data\CPU_log_infos.csv
        The command line executed here is the cli created by the setup_cli function.
        """
        returncode = None
        if self._system.startswith("win"):
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

IntelPowerGadget()
X = np.random.randn(40000, 5000)
y = np.random.randn(40000)
reg = LinearRegression().fit(X, y)