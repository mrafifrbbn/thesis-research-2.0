import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv()

# Create logging instance
logger = get_logger('fit_mocks')

ROOT_PATH = os.environ.get('ROOT_PATH')
# SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))

# Cosmological constants
omega_m = 0.3121
mag_low = 8.0
mag_high = 13.65
zmin = 3000.0 / LIGHTSPEED
zmax = 16120. / LIGHTSPEED

# Input variables
MOCK_FITS_METHOD = int(sys.argv[1])
SURVEY = sys.argv[2]
# Input filepath
MOCK_FILEPATH = sys.argv[3]
# SMIN_SETTING
SMIN_SETTING = int(sys.argv[4])

if len(sys.argv) == 6:
    OUTPUT_FILENAME = sys.argv[5]
else:
    OUTPUT_FILENAME = SURVEY.lower()
    
# # Filepaths
# MOCK_FILEPATH = os.path.join(ROOT_PATH, f'src/mocks/GENRMOCKFP_{SURVEY}/smin_setting_{SMIN_SETTING}_mock.txt')

MOCK_FIT_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/mock_fits/method_{MOCK_FITS_METHOD}/smin_setting_{SMIN_SETTING}/{OUTPUT_FILENAME}.csv')
create_parent_folder(MOCK_FIT_OUTPUT_FILEPATH)

# Velocity dispersion lower limit
smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][SURVEY]
Nfits = 500

def method_1():
    '''
    The original method to fit the mocks (no outlier rejection in the sample).
    '''
    FP_mocks = []
    for i in range(Nfits):
        data = pd.read_csv(MOCK_FILEPATH, delim_whitespace=True)
        data['mock_ID'] = data['#mockgal_ID'].map(lambda x: int(x.split('_')[1]))
        data = data[data['mock_ID']== i + 1]

        # The range of the FP parameters' values allowed
        avals, bvals = (1.1, 1.7), (-1.0, -0.5)
        rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.5), (3.0, 3.5)
        s1vals, s2vals, s3vals = (0., 0.3), (0.1, 0.5), (0.1, 0.3)
        Sn = data['Sprob'].to_numpy()
        FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
            args=(0.0, data["cz"].to_numpy()/LightSpeed, data["r"].to_numpy(), data["s"].to_numpy(), data["i"].to_numpy(), data["dr"].to_numpy(), data["ds"].to_numpy(), data["di"].to_numpy(), Sn, smin), maxiter=10000, tol=1.0e-6)
        chi_squared = Sn*FP_func(FPparams.x, 0.0, data["cz"].to_numpy()/LightSpeed, data["r"].to_numpy(), data["s"].to_numpy(), data["i"].to_numpy(), data["dr"].to_numpy(), data["ds"].to_numpy(), data["di"].to_numpy(), Sn, smin, sumgals=False, chi_squared_only=True)[0]

        # Store the FP fits
        FP_mocks.append(FPparams.x)
        time.sleep(0.1)

    # Save fits as csv
    FP_mocks = np.array(FP_mocks)
    mocks_df = pd.DataFrame(FP_mocks).rename(columns={0: 'a', 1: 'b', 2: 'rmean', 3: 'smean', 4: 'imean', 5: 'sigma1', 6: 'sigma2', 7: 'sigma3'})
    mocks_df.to_csv(MOCK_FIT_OUTPUT_FILEPATH, index=False)

def method_2():
    '''
    The latest method to fit the mocks (applying outlier rejection in the sample).
    '''
    # p-value upper limit, reject galaxies with p-value lower than this limit
    pvals_cut = 0.01

    # Load the mock file
    mocks = pd.read_csv(MOCK_FILEPATH, delim_whitespace=True)
    mocks['mock_no'] = mocks['#mockgal_ID'].apply(lambda x: int(x.split('_')[1]))
    mocks['z'] = mocks['cz'] / LightSpeed

    FP_mocks = []
    for mock_no in range(1, NFITS + 1):
        print(f'Iteration: {mock_no}/{NFITS}')
        df = mocks[mocks['mock_no'] == mock_no]
        Sn = df['Sprob'].to_numpy()

        # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
        data_fit = df
        badcount = len(df)
        is_converged = False
        i = 1

        while not is_converged:
            Snfit = data_fit['Sprob'].to_numpy()

            # The range of the FP parameters' values
            avals, bvals = (1.0, 1.8), (-1.0, -0.5)
            rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.5), (3.0, 3.5)
            s1vals, s2vals, s3vals = (0., 0.3), (0.1, 0.5), (0.1, 0.3)

            # Fit the FP parameters
            FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit["z"].to_numpy(), data_fit["r"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin), maxiter=10000, tol=1.0e-6)
            # Calculate the chi-squared 
            chi_squared = Sn*FP_func(FPparams.x, 0.0, df["z"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["dr"].to_numpy(), df["ds"].to_numpy(), df["di"].to_numpy(), Sn, smin, sumgals=False, chi_squared_only=True)[0]

            # Calculate the p-value (x,dof)
            pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
            # Reject galaxies with p-values < pvals_cut
            data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
            # Count the number of rejected galaxies
            badcountnew = len(np.where(pvals < pvals_cut)[0])
            # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
            is_converged = True if badcount == badcountnew else False

            # Set the new count of rejected galaxies
            badcount = badcountnew
            i += 1

            time.sleep(0.5)

        # Store the FP parameters
        FP_mocks.append(FPparams.x)
        FP_mocks = np.array(FP_mocks)
        mocks_df = pd.DataFrame(FP_mocks).rename(columns={0: 'a', 1: 'b', 2: 'rmean', 3: 'smean', 4: 'imean', 5: 'sigma1', 6: 'sigma2', 7: 'sigma3'})
        mocks_df.to_csv(MOCK_FIT_OUTPUT_FILEPATH, index=False)

def main():
    logger.info(f'Fitting {Nfits} mocks of {SURVEY} | MOCK_FITS_METHOD = {MOCK_FITS_METHOD} | SMIN_SETTING = {SMIN_SETTING} | OUTPUT_FILENAME = {OUTPUT_FILENAME}')
    start = time.time()
    if int(MOCK_FITS_METHOD) == 1:
        method_1()
    elif int(MOCK_FITS_METHOD) == 2:
        method_2()
    logger.info(f'Fitting mocks successful! Time elapsed = {round(time.time() - start, 2)} s.')

if __name__ == '__main__':
    main()
    # print((sys.argv))