import os
import sys
import numpy as np
import pandas as pd
from typing import Dict

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.utils.constants import *
from main_code.utils.filepaths import (
    OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
    FP_FIT_ABC_FIXED_FILEPATH,
    MCMC_CHAIN_ABC_FIXED_FILEPATH,
    MCMC_CHAIN_ABC_FIXED_CLEANED_FILEPATHS,
    FP_FIT_TYPICAL_SCATTER_FILEPATH
)
from main_code.utils.logging_config import get_logger

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create logging instance
logger = get_logger('fit_logdist')


# Constant
BURN_IN = 3200


def clean_mcmc_chain(
        raw_chain_filepath: str,
        burn_in: int,
        clean_chain_filepath: str,
) -> None:

    # Load raw chain
    params = np.load(raw_chain_filepath)

    # Filter the converged chain
    params_clean = params[burn_in:, :]

    # Save the cleaned chain
    np.save(clean_chain_filepath["FULL"], params_clean)

    # Unpack the chain and save for each survey
    params = params_clean.T
    a, b, c = params[0], params[1], params[2]
    smean_6df, imean_6df, sigma1_6df, sigma2_6df, sigma3_6df = params[3], params[4], params[5], params[6], params[7]
    smean_sdss, imean_sdss, sigma1_sdss, sigma2_sdss, sigma3_sdss = params[8], params[9], params[10], params[11], params[12]
    smean_lamost, imean_lamost, sigma1_lamost, sigma2_lamost, sigma3_lamost = params[13], params[14], params[15], params[16], params[17]

    rmean_6df = c + a * smean_6df + b * imean_6df
    rmean_sdss = c + a * smean_sdss + b * imean_sdss
    rmean_lamost = c + a * smean_lamost + b * imean_lamost

    params_6df = np.array([a, b, c, rmean_6df, smean_6df, imean_6df, sigma1_6df, sigma2_6df, sigma3_6df]).T
    params_sdss = np.array([a, b, c, rmean_sdss, smean_sdss, imean_sdss, sigma1_sdss, sigma2_sdss, sigma3_sdss]).T
    params_lamost = np.array([a, b, c, rmean_lamost, smean_lamost, imean_lamost, sigma1_lamost, sigma2_lamost, sigma3_lamost]).T

    # Save
    np.save(clean_chain_filepath["6dFGS"], params_6df)
    np.save(clean_chain_filepath["SDSS"], params_sdss)
    np.save(clean_chain_filepath["LAMOST"], params_lamost)

    return
    

def fp_fit_summary(
        chain_filepath: str,
        fp_fits_filepath: str
):
    params = np.load(chain_filepath).T
    
    param_names = [
        "a", "b", "c",
        "smean_6df", "imean_6df", "sigma1_6df", "sigma2_6df", "sigma3_6df",
        "smean_sdss", "imean_sdss", "sigma1_sdss", "sigma2_sdss", "sigma3_sdss",
        "smean_lamost", "imean_lamost", "sigma1_lamost", "sigma2_lamost", "sigma3_lamost",
    ]

    # Summary of FP fits taken as mean of the MCMC chain
    param_summary = []
    for i in range(18):

        # Fit with Gaussian
        xdata = params[i]

        param_summary.append(np.mean(xdata))

    # Summary for each survey
    param_6df = [param_summary[0], param_summary[1], param_summary[2], param_summary[3], param_summary[4], param_summary[5], param_summary[6], param_summary[7]]
    param_sdss = [param_summary[0], param_summary[1], param_summary[2], param_summary[8], param_summary[9], param_summary[10], param_summary[11], param_summary[12]]
    param_lamost = [param_summary[0], param_summary[1], param_summary[2], param_summary[13], param_summary[14], param_summary[15], param_summary[16], param_summary[17]]

    # Save as dataframe
    col_names = ["a", "b", "c", "smean", "imean", "s1", "s2", "s3"]
    param_all = [param_6df, param_sdss, param_lamost]
    df = pd.DataFrame(param_all, index=["6dFGS", "SDSS", "LAMOST"], columns=col_names)

    # Calculate rmean
    df["rmean"] = df["c"] + df["a"] * df["smean"] + df["b"] * df["imean"]
    df = df[["a", "b", "c", "rmean", "smean", "imean", "s1", "s2", "s3"]]

    df.to_csv(fp_fits_filepath)
    

def calculate_fp_scatter(
        data_filepaths: Dict[str, str],
        fp_fits_filepath: str,
        fp_scatter_destination_filepath: str
) -> None:
    results = []
    for survey in SURVEY_LIST:
        df = pd.read_csv(data_filepaths[survey])
        params = pd.read_csv(fp_fits_filepath, index_col=0).loc[survey].to_dict()
        a = params.get("a")
        b = params.get("b")
        sigma_1 = params.get("s1")
        
        # Calculate typical scatter in s
        err_spectro = df['es'].median()
        
        # Calculate the combined photometric error
        e_XFP = np.sqrt(df['er']**2 + (b * df['ei'])**2 + 2 * (-1) * np.absolute(b * df['er'] * df['ei']))
        err_photo = np.median(e_XFP)
        
        # Calculate the total intrinsic error in r
        sigma_r_int = sigma_1 * np.sqrt(1 + a**2 + b**2)

        # Calculate the observational error in r
        sigma_r_obs = np.sqrt((a * err_spectro)**2 + err_photo**2)
        
        # Calculate the total typical scatter in r
        r_scatter = np.sqrt(sigma_r_obs**2 + sigma_r_int**2)
        r_scatter_pct = np.log(10) * r_scatter * 100
        
        # Save everything in a dictionary
        scatter_dict = {
            "eps_s": err_spectro,
            "eps_photo": err_photo,
            "sigma_r_int": sigma_r_int,
            "sigma_r_obs": sigma_r_obs,
            "r_scatter": r_scatter,
            "r_scatter_pct": r_scatter_pct
        }
        results.append(scatter_dict)
        
    df = pd.DataFrame(results, index=SURVEY_LIST).round(decimals=4)

    df.to_csv(fp_scatter_destination_filepath, index=True)


def main():
    try:
        logger.info(f"{'=' * 50}")
        logger.info("Generating FP summary...")

        # 1. Clean MCMC chain (set burn-in)
        logger.info(f"Cleaning MCMC chain. Burn-in: {BURN_IN}")
        clean_mcmc_chain(
            raw_chain_filepath=MCMC_CHAIN_ABC_FIXED_FILEPATH,
            burn_in=BURN_IN,
            clean_chain_filepath=MCMC_CHAIN_ABC_FIXED_CLEANED_FILEPATHS
        )

        # 2. Calculate FP fits estimates (mean) from MCMC chain
        logger.info("Calculating mean of the chains...")
        fp_fit_summary(
            chain_filepath=MCMC_CHAIN_ABC_FIXED_CLEANED_FILEPATHS["FULL"],
            fp_fits_filepath=FP_FIT_ABC_FIXED_FILEPATH
        )

        # 3. Calculate typical scatter
        logger.info("Calculating typical FP scatter...")
        calculate_fp_scatter(
            data_filepaths=OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
            fp_fits_filepath=FP_FIT_ABC_FIXED_FILEPATH,
            fp_scatter_destination_filepath=FP_FIT_TYPICAL_SCATTER_FILEPATH
        )

        logger.info('Generating FP summary successful!')
    
    except Exception as e:
        logger.error(f'Generating FP summary failed. Reason: {e}.', exc_info=True)


if __name__ == '__main__':
    main()