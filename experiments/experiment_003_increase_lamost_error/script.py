import os
import sys
import trace
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib import cm, ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Tuple

root_dir = os.environ.get("ROOT_PATH")
if not root_dir in sys.path: sys.path.append(root_dir)

from src.step_7_fit_fp import fit_FP, sample_likelihood
from src.utils.constants import *
from src.utils.CosmoFunc import *
from src.utils.logging_config import get_logger

from src.filepaths import *

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

import emcee
import getdist

from dotenv import load_dotenv
load_dotenv(override=True)

# Get environment variables from .env file
ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
COMPLETENESS_SETTING = int(os.environ.get('COMPLETENESS_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Add new data combinations here
NEW_SURVEY_LIST = SURVEY_LIST + ['SDSS_LAMOST', 'ALL_COMBINED'] if SMIN_SETTING == 1 else SURVEY_LIST

# Create logging instance
logger = get_logger('fit_fp')

# Set global random seed
np.random.seed(42)

### CONSTANTS ###
N_list = [5, 10, 15, 20]

def main() -> None:

    # Load LAMOST data
    input_filepath = FOUNDATION_ZONE_FP_SAMPLE_FILEPATHS["LAMOST"]
    df = pd.read_csv(input_filepath)

    # Loop over different N values
    FP_params = []
    for N in N_list:

        # Increase LAMOST veldisp errors by N per cent
        df['es'] = df['es'] * (1 + N / 100)

        # Veldisp lower limit
        smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING]["LAMOST"]

        # FP parameter boundaries
        param_boundaries = PARAM_BOUNDARIES

        # Fit the FP
        params, df_fitted = fit_FP(
                survey="LAMOST",
                df=df,
                smin=smin,
                param_boundaries=param_boundaries,
                reject_outliers=True,
                use_full_fn=True
            )
        FP_params.append(params)

        # Sample the likelihood using MCMC
        chain_output_filepath = f"lamost_error_increased_by_{N}_percent.npy"
        params_mean = sample_likelihood(
            df=df_fitted,
            FP_params=params,
            smin=smin,
            chain_output_filepath=chain_output_filepath
            )

    # Convert the FP parameters to dataframe and save to artifacts folder
    FP_params = np.array(FP_params)
    FP_columns = ['a', 'b', 'rmean', 'smean', 'imean', 's1', 's2', 's3']
    pd.DataFrame(FP_params, columns=FP_columns, index=[f"LAMOST_{N}" for N in N_list]).to_csv("fp_fits.csv")

if __name__ == '__main__':
    main()