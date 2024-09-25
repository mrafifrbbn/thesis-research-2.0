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
MOCK_FITS_METHOD = int(sys.argv[1]) # Mock fits method: 1 for no outlier rejection, 2 for outlier rejection
SURVEY = sys.argv[2] # Survey name = ['6dFGS', 'SDSS', 'LAMOST']
# Input filepath
MOCK_FILEPATH = sys.argv[3] # Mock data path (output from Christina's mock generation code)
# SMIN_SETTING
SMIN_SETTING = int(sys.argv[4]) # smin_setting (should be always 1 now)

if len(sys.argv) == 6:
    OUTPUT_FILENAME = sys.argv[5] # Output filename (excluding path and csv)
else:
    OUTPUT_FILENAME = SURVEY.lower()
    
# # Filepaths
# MOCK_FILEPATH = os.path.join(ROOT_PATH, f'src/mocks/GENRMOCKFP_{SURVEY}/smin_setting_{SMIN_SETTING}_mock.txt')

MOCK_FIT_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/mock_fits/method_{MOCK_FITS_METHOD}/smin_setting_{SMIN_SETTING}/{OUTPUT_FILENAME}.txt')

# Velocity dispersion lower limit
smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][SURVEY]

# Start and end ID of the mock (change if something fails midway)
ID_START = 1
ID_END = 200

def main():
    logger.info(f'Fitting {ID_END - ID_START + 1} mocks of {SURVEY} | MOCK_FITS_METHOD = {MOCK_FITS_METHOD} | SMIN_SETTING = {SMIN_SETTING} | OUTPUT_FILENAME = {OUTPUT_FILENAME}')

    # Create output file for mock fits if not exist (otherwise append to last line)
    if not os.path.exists(MOCK_FIT_OUTPUT_FILEPATH):
        with open(MOCK_FIT_OUTPUT_FILEPATH, "w") as fp:
            fp.write("mock_id,a,b,rmean,smean,imean,sigma1,sigma2,sigma3\n")
            pass

    # Load the mock galaxies data
    mock = pd.read_csv(MOCK_FILEPATH, delim_whitespace=True)

    # Maximum veldisp from the data
    s_max = pd.read_csv(f'{ROOT_PATH}/data/foundation/fp_sample_final/smin_setting_1/{SURVEY.lower()}.csv')['s'].max()

    start = time.time()
    for mock_id in range(ID_START, ID_END + 1):
        df = mock.copy()

        # Select one mock realization at a time to fit
        df = df[df['#mockgal_ID'].apply(lambda x: int(x.split('_')[-1])) == mock_id]
        # # Apply max veldisp limit
        # df = df[df['s'] <= s_max]
        df['z'] = df['cz'] / LIGHTSPEED

        # Calculate predicted true distance and FN integral limits
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        d_H = sp.interpolate.splev(df['z'].to_numpy(), dist_spline, der=0)
        df['lmin'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_HIGH) / 5.0
        df['lmax'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["z"].to_numpy()) + df["kcorr"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0

        # Calculate predicted logdistance-ratios
        d_z = sp.interpolate.splev(df['z'].to_numpy(), dist_spline, der=0)
        df['logdist_pred'] = np.log10(d_z / d_H)
        df['r_true'] = df['r'] - df['logdist_pred']

        # Use Sn (default is not)
        if False: # Replace this with a variable in the future
            # Get some redshift-distance lookup tables
            red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
            # The comoving distance to each galaxy using group redshift as distance indicator
            dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0)

            # (1+z) factor because we use luminosity distance
            Vmin = (1.0 + zmin)**3 * sp.interpolate.splev(zmin, dist_spline)**3
            Vmax = (1.0 + zmax)**3 * sp.interpolate.splev(zmax, dist_spline)**3
            # Maximum (luminosity) distance the galaxy can be observed given MAG_HIGH (survey limiting magnitude)
            Dlim = 10.0**((mag_high - (df["j_m_ext"] - df['extinction_j']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)    
            # Find the corresponding maximum redshift
            zlim = sp.interpolate.splev(Dlim, lumred_spline)
            Sn = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))
            df['Sn'] = Sn
        else:
            Sn = df['Sprob'].to_numpy() + 1.0e-15
            df['Sn'] = Sn
            df['C_m'] = 1.0

        # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
        data_fit = df
        badcount = len(df)
        is_converged = False
        pvals_cut = 0.01

        while not is_converged:

            Snfit = data_fit["Sn"].to_numpy()

            # The range of the FP parameters' values
            param_boundaries = [(1.2, 1.6), (-0.9, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.2, 3.5), (0.0, 0.06), (0.25, 0.45), (0.10, 0.25)]
            avals, bvals = param_boundaries[0], param_boundaries[1]
            rvals, svals, ivals = param_boundaries[2], param_boundaries[3], param_boundaries[4]
            s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

            # Fit the FP parameters
            FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit["z"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, False), maxiter=10000, tol=1.0e-6, workers=-1)
            # # Calculate the chi-squared 
            # chi_squared = Sn * FP_func(FPparams.x, 0.0, df["z"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["dr"].to_numpy(), df["ds"].to_numpy(), df["di"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, chi_squared_only=True, use_full_fn=True)[0]
            # # Calculate the p-value (x,dof)
            # pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
            # # Reject galaxies with p-values < pvals_cut (probabilities of being part of the sample lower than some threshold)
            # data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
            # # Count the number of rejected galaxies
            # badcountnew = len(np.where(pvals < pvals_cut)[0])
            # # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
            # is_converged = True if badcount == badcountnew else False

            # # logger.info verbose
            print(f"{'-' * 10} Mock {mock_id} {'-' * 10} | FP parameters: {FPparams.x.tolist()}")

            # Break from the loop if reject_outliers is set to false
            if MOCK_FITS_METHOD == 1:
                break

        df = data_fit
        with open(MOCK_FIT_OUTPUT_FILEPATH, "a") as myfile:
            text = ','.join([str(mock_id)] + [str(x) for x in FPparams.x]) + '\n'
            myfile.write(text)
    logger.info(f'Fitting mocks successful! Time elapsed = {round(time.time() - start, 2)} s.')

if __name__ == '__main__':
    main()
    # print((sys.argv))