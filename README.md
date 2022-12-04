# SnakeChamers
Machine learning for predicting energy consumption and cost when running code.




## Environment requirements

The packages necessary for the environment setup are specified in *requirements/* where you can chose the file for either macOS or Windows. The main packages are **codecarbon** which uses information from the Intel processor to estimate the energy consumption during python code execution.

## Repository organisation
The directory *data* contains all the CSV files obtained during our pipeline training. The python files *hardware_features extractor.py* and *energy_extractor_intel.py* are utilitary files that are need for running our main pathon script *dataset_generator*

## Supplementary info
To test a model. e.g. LinearRegression
```python
from sklean.linear_model import LinearRegression
```

## Results

Below we have uploaded visualizations of our dataset generated using the *dataset_generator.py* pipeline.

<p align="center">
<img src="./data/img/img1.png" alt="Training curve over 15 000 steps" width="45%"/>
<img src="./data/img/img1.png" alt="Best score" width="45%"/>
</p>

We have also plotted the predictions of a LinearRegressor fitted on our dataset.

We can see that our models .... predicts the energy consumption.
<p align="center">
<img src="./data/img/img1.png" alt="Training curve over 15 000 steps"/>
</p>

