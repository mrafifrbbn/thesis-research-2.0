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

from main_code.utils.constants import *
from main_code.utils.functions import gaus
from main_code.utils.CosmoFunc import *
from main_code.utils.filepaths import (
    OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
    FP_FIT_FILEPATH,
    FP_FIT_ABC_FIXED_FILEPATH
)
from main_code.utils.logging_config import get_logger

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Create logging instance
logger = get_logger('fit_logdist')


def fit_logdist(
        survey: str,
        df: pd.DataFrame,
        smin: float,
        FPmethod: str,
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
    dz = sp.interpolate.splev(df["z_cmb"].to_numpy() / LIGHTSPEED, dist_spline)
    dz_cluster = sp.interpolate.splev(df["z_dist_est"], dist_spline)
    logger.info(f'Number of {survey} data = {len(df)}.')

    # Define the range of logdists values to be calculated
    dmin, dmax, nd = -1.5, 1.5, 1001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    pv_model = TwoMPP_SDSS_6dF(verbose=True) # type: ignore

    # Calculate predicted PVs using observed group redshift in CMB frame, and calculate cosmological redshift
    df['v_pec'] = pv_model.calculate_pv(df['ra'].to_numpy(), df['dec'].to_numpy(), df['z_dist_est'].to_numpy())
    df['z_pec'] = df['v_pec'] / LIGHTSPEED
    df['z_cosmo'] = ((1 + df['z_dist_est']) / (1 + df['z_pec'])) - 1
    d_H_pred = sp.interpolate.splev(df['z_cosmo'].to_numpy(), dist_spline, der=0)

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    
    # Calculate predicted logdistance-ratios (subtract with prediction from PV model)
    d_z = sp.interpolate.splev(df['z_dist_est'].to_numpy(), dist_spline, der=0)
    df['logdist_pred'] = np.log10(d_z / d_H_pred)
    df['r_true'] = df['r'] - df['logdist_pred']

    # Calculate negative of log likelihood
    loglike = FP_func(FPparams, dbins, df["z_cmb"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), np.ones(len(df)), smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, use_full_fn=use_full_fn)
    
    # Calculate full FN
    FNvals = FN_func(FPparams, df["z_cmb"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), lmin, lmax, smin)

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
    df[f"logdist_mean_{FPmethod.lower()}"] = mean
    df[f"logdist_std_{FPmethod.lower()}"] = err
    df[f"logdist_alpha_{FPmethod.lower()}"] = alpha
    df[f"logdist_loc_{FPmethod.lower()}"] = loc
    df[f"logdist_scale_{FPmethod.lower()}"] = scale

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)
    # Save the posterior distributions
    if save_posterior:
        logdist_posterior_filepath = os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/{survey.lower()}_posterior.npy')
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

    df[f'logdist_{FPmethod.lower()}'] = logdist_mean
    df[f'logdist_err_{FPmethod.lower()}'] = logdist_std

    # Calculate observational error
    a, b, sigma1 = FPparams[0], FPparams[1], FPparams[5]
    logdist_int_err = sigma1 * np.sqrt(1 + a**2 + b**2)
    df[f"logdist_int_err_{FPmethod.lower()}"] = logdist_int_err
    df[f"logdist_obs_err_{FPmethod.lower()}"] = np.sqrt(np.array(logdist_std)**2 - logdist_int_err**2)

    # Calculate logdist fit's goodness-of-fit
    df[f'logdist_fit_chisq_{FPmethod.lower()}'] = chisq
    df[f'logdist_fit_rmse_{FPmethod.lower()}'] = rmse
    
    return df


def main():
    try:
        logger.info(f"{'=' * 50}")
        logger.info('Fitting log-distance ratios...')
        logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}.')
        
        # FP base parameters
        fp_base_params = ["a", "b", "rmean", "smean", "imean", "s1", "s2", "s3"]

        # Load FP parameters for fitting
        fp_methods_dict = {
            "individual": {},
            "combined": {},
            "abc_fixed": {}
        }
        # 1. Individual and combined FP
        fp_params = pd.read_csv(FP_FIT_FILEPATH, index_col=0)
        for survey in SURVEY_LIST:
            fp_methods_dict["individual"][survey] = fp_params.loc[survey][fp_base_params].to_numpy()
            fp_methods_dict["combined"][survey] = fp_params.loc["ALL_COMBINED"][fp_base_params].to_numpy()

        # 2. abc-fixed FP
        fp_params = pd.read_csv(FP_FIT_ABC_FIXED_FILEPATH, index_col=0)
        for survey in SURVEY_LIST:
            fp_methods_dict["abc_fixed"][survey] = fp_params.loc[survey][fp_base_params].to_numpy()

        for survey in SURVEY_LIST:
            print(survey)
            # Get input filename (outlier-rejected sample)
            input_filepath = OUTLIER_REJECT_FP_SAMPLE_FILEPATHS[survey]
            df = pd.read_csv(input_filepath)
            
            # Survey's veldisp limit
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]

            # Derive logdists for all FP methods
            for fp_method in fp_methods_dict:
                params = fp_methods_dict[fp_method][survey]
                df = fit_logdist(
                    survey=survey,
                    df=df,
                    smin=smin,
                    FPmethod=fp_method,
                    FPparams=params,
                    use_full_fn=USE_FULL_FN,
                    save_posterior=True
                    )

            # Save logdist measurements
            logdist_output_filepath = os.path.join(ROOT_PATH, f'experiments/experiment_009_logdist_pv_model/{survey}.csv')
            df.to_csv(logdist_output_filepath, index=False)

        logger.info('Fitting log-distance ratios successful!')
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.', exc_info=True)


if __name__ == '__main__':
    main()
