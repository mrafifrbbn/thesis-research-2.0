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
# Add new data combinations here
NEW_SURVEY_LIST = (SURVEY_LIST + ['6dFGS_SDSS', 'SDSS_LAMOST', 'ALL_COMBINED']) if SMIN_SETTING == 1 else SURVEY_LIST

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Create logging instance
logger = get_logger('fit_logdist')


def fit_logdist(
        survey: str,
        df: pd.DataFrame,
        smin: float,
        FPlabel: str,
        FPparams: np.ndarray,
        use_full_fn: bool = True,
        mag_high: float = MAG_HIGH,
        mag_low: float = MAG_LOW,
        save_posterior: bool = False
    ) -> pd.DataFrame:
    """
    This is a function to calculate the log-distance ratio posteriors and obtain summary statistics.
    Summary statistics are obtained using two methods: direct calculation assuming skew-normal and 
    fitting using scipy's curve_fit assuming Gaussian.
    """
    logger.info(f"Calculating the logdistance PDF for galaxies in {survey} | FP: {list(FPparams)}")

    # Get some redshift-distance lookup tables
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    dz = sp.interpolate.splev(df["z_cmb"].to_numpy() / LightSpeed, dist_spline)
    dz_cluster = sp.interpolate.splev(df["z_dist_est"], dist_spline)
    logger.info(f'Number of {survey} data = {len(df)}.')

    # Define the range of logdists values to be calculated
    dmin, dmax, nd = -1.5, 1.5, 2001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    # if survey.lower() == "sdss":
    #     df["es"] *= 1.5

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    loglike = FP_func(FPparams, dbins, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), np.ones(len(df)), smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, use_full_fn=use_full_fn)
    
    # Calculate full FN
    FNvals = np.log(0.6) #FN_func(FPparams, df["z_cmb"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), lmin, lmax, smin)

    # Convert to the PDF for logdistance
    logP_dist = -1.5 * np.log(2.0 * math.pi) - loglike - FNvals

    # normalise logP_dist (trapezoidal rule)
    ddiff = np.log10(d_H[:-1]) - np.log10(d_H[1:])
    valdiff = np.exp(logP_dist[1:]) + np.exp(logP_dist[0:-1])
    norm_ = 0.5 * np.sum(valdiff * ddiff, axis=0)

    logP_dist -= np.log(norm_[:, None]).T

    # Calculate the mean and standard deviation of the distribution
    mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0
    err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2)

    # Calculate the skewness and cap it
    gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
    gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
    gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)

    # Calculate the parameters of skew-normal distribution
    delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
    scale = err * np.sqrt(1.0 / (1.0 - 2.0 * delta**2 / np.pi))
    loc = mean - scale * delta * np.sqrt(2.0 / np.pi)
    alpha = delta / (np.sqrt(1.0 - delta**2))

    # Store the skew-normal values calculated analytically to the dataframe
    df[f"logdist_mean_{FPlabel.lower()}"] = mean
    df[f"logdist_std_{FPlabel.lower()}"] = err
    df[f"logdist_alpha_{FPlabel.lower()}"] = alpha
    df[f"logdist_loc_{FPlabel.lower()}"] = loc
    df[f"logdist_scale_{FPlabel.lower()}"] = scale

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)
    # Save the posterior distributions
    if save_posterior:
        logdist_posterior_filepath = os.path.join(ROOT_PATH, f'experiments/experiment_006_logdist_no_fn/{survey.lower()}_pdf.npy')
        np.save(logdist_posterior_filepath, y)

    # Find mean and standard deviation of the distribution using curve_fit
    logdist_mean = []
    logdist_std = []
    chisq = []
    rmse = []
    for i, y_ in enumerate(y):
        popt, pcov = curve_fit(gaus, dbins, y_, p0=[mean[i], err[i]])
        popt[1] = np.absolute(popt[1])
        logdist_mean.append(popt[0])
        logdist_std.append(popt[1])

        # Calculate the chi-squared statistic for the fit
        ypred = norm.pdf(dbins, popt[0], popt[1])
        chisquare = np.sum((y_ - ypred)**2 / ypred, axis=0)
        chisq.append(chisquare)

        # Calculate RMSE statistic for the fit
        ypred = norm.pdf(dbins, popt[0], popt[1])
        rmse_ = np.sqrt((1 / dbins.shape[0]) * np.sum((y_ - ypred)**2, axis=0))
        rmse.append(rmse_)

    df[f'logdist_{FPlabel.lower()}'] = logdist_mean
    df[f'logdist_err_{FPlabel.lower()}'] = logdist_std
    df[f'logdist_chisq_{FPlabel.lower()}'] = chisq
    df[f'logdist_rmse_{FPlabel.lower()}'] = rmse
    
    return df


def main():
    try:
        logger.info(f"{'=' * 50}")
        logger.info('Fitting log-distance ratios...')
        logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}.')
        
        for survey in SURVEY_LIST:
            print(survey)
            # Get input filename (outlier-rejected sample)
            input_filepath = OUTLIER_REJECT_FP_SAMPLE_FILEPATHS[survey]
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
                    use_full_fn=USE_FULL_FN,
                    save_posterior=True
                    )

            # Save logdist measurements
            logdist_output_filepath = os.path.join(ROOT_PATH, f"experiments/experiment_006_logdist_no_fn/{survey.lower()}.csv")
            df.to_csv(logdist_output_filepath, index=False)

        logger.info('Fitting log-distance ratios successful!')
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.')


if __name__ == '__main__':
    main()