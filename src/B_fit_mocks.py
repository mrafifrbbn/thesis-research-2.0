import os
import sys
import re
import time
import numpy as np
import pandas as pd
import scipy as sp
import argparse
from pathlib import Path

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv(override=True)

# Create logging instance
logger = get_logger('fit_mocks')

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from src.filepaths import *
from src.utils.constants import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--id-start", help="Mock ID to start fitting over.", type=int, default=1
)
parser.add_argument(
    "--id-end", help="Mock ID to end fitting over.", type=int, default=1000
)
parser.add_argument(
    "--fresh-start", help="Deletes the output file (if exists) if called.", action="store_true"
)

# Start and end ID of the mock (change if something fails midway)
args = parser.parse_args()
ID_START = args.id_start
ID_END = args.id_end

def fit_mock(
        mock_filename: str,
        survey: str,
        fp_fit_method: int,
        smin_setting: int,
        id_start: int,
        id_end: int,
        output_filepath: str,
        reject_outliers: bool = False,
        pvals_cut: float = 0.01
        ):
    """_summary_

    Args:
        mock_filename (str): _description_
        survey (str): _description_
        fp_fit_method (int): _description_
        smin_setting (int): _description_
        id_start (int): _description_
        id_end (int): _description_
        output_filepath (str): _description_
    """
    
    logger.info(f'Fitting {id_end - id_start + 1} mocks of {survey} | FP Fit Method = {fp_fit_method} | SMIN_SETTING = {smin_setting}')

    # Create boolean to store whether to use full f_n or partial f_n
    use_full_fn = True if fp_fit_method == 0 else False

    # Set veldisp lower limit
    smin = SURVEY_VELDISP_LIMIT[smin_setting][survey]

    # Load the mock galaxies data
    mock = pd.read_csv(mock_filename, delim_whitespace=True)

    start = time.time()
    for mock_id in range(id_start, id_end + 1):
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

        if not use_full_fn:
            Sn = df["Sprob"].to_numpy()
        else:
            Sn = 1.0

        df['Sn'] = Sn
        df['C_m'] = 1.0

        # Range of FP parameters
        if fp_fit_method == 0:
            param_boundaries = PARAM_BOUNDARIES
        else:
            param_boundaries = [(1.2, 1.8), (-1.1, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.0, 0.06), (0.20, 0.45), (0.1, 0.25)]

        avals, bvals = param_boundaries[0], param_boundaries[1]
        rvals, svals, ivals = param_boundaries[2], param_boundaries[3], param_boundaries[4]
        s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

        # Fitting the FP iteratively
        data_fit = df
        badcount = len(df)
        is_converged = False
        i = 1
        while not is_converged:

            Snfit = data_fit["Sn"].to_numpy()

            # Fit the FP parameters
            FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit["z"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, use_full_fn), maxiter=10000, tol=1.0e-6, workers=-1)

            # Break from the loop if not iterative
            if reject_outliers == True:
                break

            # Calculate the chi-squared 
            chi_squared = Sn * FP_func(FPparams.x, 0.0, data_fit["z"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["dr"].to_numpy(), data_fit["ds"].to_numpy(), data_fit["di"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), sumgals=False, chi_squared_only=True, use_full_fn=use_full_fn)[0]
            
            # Calculate the p-value (x,dof)
            pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(data_fit) - 8.0))
            
            # Reject galaxies with p-values < pvals_cut (probabilities of being part of the sample lower than some threshold)
            data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
            
            # Count the number of rejected galaxies
            badcountnew = len(np.where(pvals < pvals_cut)[0])
            
            # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
            is_converged = True if badcount == badcountnew else False
            
            # Set the new count of rejected galaxies
            badcount = badcountnew
            i += 1
            
            # Break from the loop if reject_outliers is set to false
            if reject_outliers == False:
                break

        # logger.info verbose
        print(f"{'-' * 10} Mock {mock_id} {'-' * 10} | FP parameters: {FPparams.x.tolist()}")

        df = data_fit
        
        # Append the results to the output file
        with open(output_filepath, "a") as myfile:
            text = ','.join([str(mock_id)] + [str(x) for x in FPparams.x]) + '\n'
            myfile.write(text)

    logger.info(f'Fitting mocks successful! Time elapsed = {round(time.time() - start, 2)} s.')
    return


def main():

    mock_filenames = os.listdir(MOCK_DATA_FILEPATH)

    for filename in mock_filenames:
        # Parse the filename to retrieve settings
        survey = re.search(r"(.*)_mocks", filename).group(1)

        if survey.lower() == '6dfgs':
            survey = '6dFGS'
        else:
            survey = survey.upper()

        smin_setting = int(re.search(r"smin_(\d+)", filename).group(1))
        fp_fit_method = int(re.search(r"fp_fit_method_(\d+)", filename).group(1))

        # Generate output file path string
        output_filepath = Path(os.path.join(ROOT_PATH, f"artifacts/mock_fits/smin_setting_{smin_setting}/fp_fit_method_{fp_fit_method}/{survey.lower()}.csv"))

        # Create parent directory for the mock fits
        if not os.path.exists(output_filepath.parent):
            os.makedirs(output_filepath.parent)

        # Deletes file if fresh_start is called
        if args.fresh_start:
            if os.path.exists(output_filepath):
                os.remove(output_filepath)
            with open(output_filepath, "w") as fp:
                fp.write("mock_id,a,b,rmean,smean,imean,sigma1,sigma2,sigma3\n")
            last_id = 1
        else:
            # Create file if not exist
            if not os.path.exists(output_filepath):
                with open(output_filepath, "w") as fp:
                    fp.write("mock_id,a,b,rmean,smean,imean,sigma1,sigma2,sigma3\n")
                    last_id = 1
            # Else, fetch the latest mock ID
            else:
                last_id = int(pd.read_csv(output_filepath)['mock_id'].max()) + 1

        fit_mock(
            mock_filename=os.path.join(MOCK_DATA_FILEPATH, filename),
            survey=survey,
            fp_fit_method=fp_fit_method,
            smin_setting=smin_setting,
            id_start=last_id,
            id_end=ID_END,
            output_filepath=output_filepath,
            reject_outliers=True
        )


if __name__ == '__main__':
    main()