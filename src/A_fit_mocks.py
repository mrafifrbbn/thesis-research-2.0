import os
import sys
import time
import numpy as np
import pandas as pd
import scipy as sp
from pathlib import Path

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fp-fit-method", help="How to fit the FP. 0 for full_fn method, 1 for partial_fn method", type=int
)
parser.add_argument(
    "--survey", help="Survey name to fit the mocks. Options: ['6dFGS', 'SDSS', 'LAMOST']", type=str
)
parser.add_argument(
    "--mock-filepath", help="Absolute filepath to the mock data (output from Christina's mock generation code).", type=str
)
parser.add_argument(
    "--smin-setting", help="Velocity dispersion lower limit. 0: using each survey's nominal velocity dispersion. 1: using 6dFGS velocity dispersion lower limit", type=int
)
# parser.add_argument(
#     "--output-filepath", help="Absolute file path to store the output (mock fits).", type=str
# )
parser.add_argument(
    "--id-start", help="Mock ID to start fitting over.", type=int, default=1
)
parser.add_argument(
    "--id-end", help="Mock ID to end fitting over.", type=int, default=500
)
parser.add_argument(
    "--fresh-start", help="Deletes the output file (if exists) if called", action="store_true"
)

args = parser.parse_args()

# Create logging instance
logger = get_logger('fit_mocks')

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))

# Input variables
FP_FIT_METHOD = args.fp_fit_method
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

SURVEY = args.survey
MOCK_FILEPATH = args.mock_filepath
SMIN_SETTING = args.smin_setting
MOCK_FIT_OUTPUT_FILEPATH = Path(os.path.join(ROOT_PATH, f'artifacts/mock_fits/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/{SURVEY.lower()}.txt'))

# Velocity dispersion lower limit
smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][SURVEY]

# Start and end ID of the mock (change if something fails midway)
ID_START = args.id_start
ID_END = args.id_end

def main():
    logger.info(f'Fitting {ID_END - ID_START + 1} mocks of {SURVEY} | FP Fit Method = {FP_FIT_METHOD} | SMIN_SETTING = {SMIN_SETTING}')

    # Deletes file if fresh_start is called
    if args.fresh_start:
        if os.path.exists(MOCK_FIT_OUTPUT_FILEPATH):
            os.remove(MOCK_FIT_OUTPUT_FILEPATH)

    # Create output file for mock fits if not exist (otherwise append to last line)
    if not os.path.exists(MOCK_FIT_OUTPUT_FILEPATH.parent):
        os.makedirs(MOCK_FIT_OUTPUT_FILEPATH.parent)
        with open(MOCK_FIT_OUTPUT_FILEPATH, "w") as fp:
            fp.write("mock_id,a,b,rmean,smean,imean,sigma1,sigma2,sigma3\n")

    # Load the mock galaxies data
    mock = pd.read_csv(MOCK_FILEPATH, delim_whitespace=True)

    start = time.time()
    for mock_id in range(ID_START, ID_END + 1):
        df = mock.copy()

        # Select one mock realization at a time to fit
        df = df[df['#mockgal_ID'].apply(lambda x: int(x.split('_')[-1])) == mock_id]

        # Calculate redshift
        df['z'] = df['cz'] / LIGHTSPEED

        # Calculate predicted true distance and FN integral limits
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        d_H = sp.interpolate.splev(df['z'].to_numpy(), dist_spline, der=0)
        df['lmin'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_HIGH) / 5.0
        df['lmax'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0

        # Assume 0 peculiar velocities
        df['logdist_pred'] = 0.0
        df['r_true'] = df['r'] - df['logdist_pred']

        if not USE_FULL_FN:
            Sn = df["Sprob"].to_numpy()
        else:
            Sn = 1.0

        df['Sn'] = Sn
        df['C_m'] = 1.0

        # Fitting the FP once (not iteratively)
        data_fit = df
        is_converged = False

        while not is_converged:

            Snfit = data_fit["Sn"].to_numpy()

            # The range of the FP parameters' values
            param_boundaries = PARAM_BOUNDARIES
            avals, bvals = param_boundaries[0], param_boundaries[1]
            rvals, svals, ivals = param_boundaries[2], param_boundaries[3], param_boundaries[4]
            s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

            # Fit the FP parameters
            FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit["z"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, USE_FULL_FN), maxiter=10000, tol=1.0e-6, workers=-1)

            # logger.info verbose
            print(f"{'-' * 10} Mock {mock_id} {'-' * 10} | FP parameters: {FPparams.x.tolist()}")

            # Break from the loop
            if True:
                break

        df = data_fit
        
        # Append the results to the output file
        with open(MOCK_FIT_OUTPUT_FILEPATH, "a") as myfile:
            text = ','.join([str(mock_id)] + [str(x) for x in FPparams.x]) + '\n'
            myfile.write(text)

    logger.info(f'Fitting mocks successful! Time elapsed = {round(time.time() - start, 2)} s.')

if __name__ == '__main__':
    main()