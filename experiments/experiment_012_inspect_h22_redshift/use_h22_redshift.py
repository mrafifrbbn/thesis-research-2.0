import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

from src.step_8_fit_logdist import fit_logdist
from src.utils.constants import *
from src.utils.CosmoFunc import *
from src.filepaths import (
    OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
    FP_FIT_FILEPATH,
    LOGDIST_POSTERIOR_OUTPUT_FILEPATH,
    LOGDIST_OUTPUT_FILEPATH,
    CURVEFIT_COMPARISON_IMG_FILEPATH,
    POSTERIOR_SKEWNESS_IMG_FILEPATH
)
from src.utils.logging_config import get_logger

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Create logging instance
logger = get_logger('fit_logdist')


def main():
    try:
        logger.info(f"{'=' * 50}")
        logger.info('Fitting log-distance ratios...')
        logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}.')
        
        for survey in ["SDSS"]:
            print(survey)
            # Get input filename (outlier-rejected sample)
            input_filepath = os.path.join(ROOT_PATH, "experiments/experiment_012_inspect_h22_redshift/sdss_h22_z.csv")
            df = pd.read_csv(input_filepath)
            
            # Survey's veldisp limit
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]

            # Iterate all available FP
            print(FP_FIT_FILEPATH)
            FPparams = pd.read_csv(FP_FIT_FILEPATH, index_col=0)

            for fp_label, params in FPparams.iterrows():
                df = fit_logdist(
                    survey=survey,
                    df=df,
                    smin=smin,
                    FPlabel=fp_label,
                    FPparams=params,
                    use_full_fn=True,
                    save_posterior=False
                    )

            # Save logdist measurements
            logdist_output_filepath = os.path.join(ROOT_PATH, f'experiments/experiment_012_inspect_h22_redshift/{survey}_logdist.csv')
            df.to_csv(logdist_output_filepath, index=False)

        logger.info('Fitting log-distance ratios successful!')
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.')


if __name__ == '__main__':
    main()
