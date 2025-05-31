import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.utils.constants import *
from main_code.utils.CosmoFunc import *
from main_code.utils.functions import gaus, skewnormal
from main_code.utils.filepaths import (
    OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
    FP_FIT_FILEPATH,
    FP_FIT_ABC_FIXED_FILEPATH,
    FP_FIT_TYPICAL_SCATTER_FILEPATH,
    LOGDIST_POSTERIOR_OUTPUT_FILEPATH,
    LOGDIST_OUTPUT_FILEPATH,
    CURVEFIT_COMPARISON_IMG_FILEPATH,
    POSTERIOR_SKEWNESS_IMG_FILEPATH
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

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    
    # Calculate negative of log likelihood
    loglike = FP_func(FPparams, dbins, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), np.ones(len(df)), smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, use_full_fn=use_full_fn)
    
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

    # Calculate observational error (subtract from intrinsic error)
    a, b, sigma1 = FPparams[0], FPparams[1], FPparams[5]

    # Calculate logdist observational error from FP spectro and photo errors
    e_XFP = np.sqrt(df['er']**2 + (b * df['ei'])**2 + 2 * (-1) * np.absolute(b * df['er'] * df['ei']))
    err_photo = e_XFP
    err_spectro = df['es']
    sigma_r_obs = np.sqrt((a * err_spectro)**2 + err_photo**2)
    typical_sigma_r_obs = np.median(sigma_r_obs)
    df[f"logdist_obs_err_nominal_{FPmethod.lower()}"] = sigma_r_obs
     
    # Calculate logdist observational error by subtracting intrinsic error
    logdist_int_err = sigma1 * np.sqrt(1 + a**2 + b**2)
    df[f"logdist_int_err_{FPmethod.lower()}"] = logdist_int_err
    logdist_obs_err_subtract = np.sqrt(np.array(logdist_std)**2 - logdist_int_err**2)
    df[f"logdist_obs_err_{FPmethod.lower()}"] = logdist_obs_err_subtract
    df[f"logdist_obs_err_{FPmethod.lower()}"] = df[f"logdist_obs_err_{FPmethod.lower()}"]#.fillna(typical_sigma_r_obs)

    # Calculate logdist fit's goodness-of-fit
    df[f'logdist_fit_chisq_{FPmethod.lower()}'] = chisq
    df[f'logdist_fit_rmse_{FPmethod.lower()}'] = rmse
    
    return df


def compare_cf_vs_formula() -> None:
    """
    This function compares the mean and std obtained using analytical formula and the one obtained from scipy's curve_fit.
    """
    for survey in NEW_SURVEY_LIST:
        df = pd.read_csv(LOGDIST_OUTPUT_FILEPATH[survey])
        
        # Compare Cullan's mean and curve_fit mean
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
        ax1.scatter(df['logdist_mean'], df['logdist_mean_cf'], s=1)
        ax1.set_ylabel('Mean from curve_fit', size=14)
        ax1.set_xlabel('Mean from formula', size=14)

        # Compare Cullan's std and curve_fit std
        ax2.scatter(df['logdist_std'], df['logdist_std_cf'], s=1)
        ax2.set_ylabel('Standard deviation from curve_fit', size=14)
        ax2.set_xlabel('Standard deviation from formula', size=14)

        plt.savefig(CURVEFIT_COMPARISON_IMG_FILEPATH[survey], dpi=300)


def plot_best_worst_posterior() -> None:
    """
    This function compares the least and most skewed posterior distributions.
    """
    for survey in NEW_SURVEY_LIST:
        # Load the posterior distributions
        df = pd.read_csv(LOGDIST_OUTPUT_FILEPATH[survey])
        dbins = np.linspace(-1.5, 1.5, 1001, endpoint=True)
        y = np.load(LOGDIST_POSTERIOR_OUTPUT_FILEPATH[survey])

        # Grab the index with the best and worst chi-square (relative to a Gaussian distribution)
        index_worst = df['logdist_chisq_cf'].sort_values().index[-1]
        index_best = df['logdist_chisq_cf'].sort_values().index[0]

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

        # Plot the best posteriors
        ax1.plot(dbins, y[index_best, :], label='Best posterior', color='k')
        ax1.plot(dbins, norm.pdf(dbins, loc=df.loc[index_best, 'logdist_mean_cf'], scale=df.loc[index_best, 'logdist_std_cf']), label='curve fit')
        ax1.plot(dbins, skewnormal(dbins, loc=df.loc[index_best, 'logdist_loc'], err=df.loc[index_best, 'logdist_scale'], alpha=df.loc[index_best, 'logdist_alpha']), c='red', ls='--', label='Skew-normal')
        ax1.set_xlim(-1.0, 1.0)
        ax1.set_xlabel(r'$\eta$', size=20)
        ax1.set_ylabel(r'PDF', size=20)
        ax1.legend(fontsize=14)

        # Plot the worst posterior
        ax2.plot(dbins, y[index_worst, :], label='Worst posterior', color='k')
        ax2.plot(dbins, norm.pdf(dbins, loc=df.loc[index_worst, 'logdist_mean_cf'], scale=df.loc[index_worst, 'logdist_std_cf']), label='curve fit')
        ax2.plot(dbins, skewnormal(dbins, loc=df.loc[index_worst, 'logdist_loc'], err=df.loc[index_worst, 'logdist_scale'], alpha=df.loc[index_worst, 'logdist_alpha']), c='red', ls='--', label='Skew-normal')
        ax2.set_xlim(-1.0, 1.0)
        ax2.set_xlabel(r'$\eta$', size=20)
        ax2.set_ylabel(r'PDF', size=20)
        ax2.legend(fontsize=14)

        plt.savefig(POSTERIOR_SKEWNESS_IMG_FILEPATH[survey], dpi=300)


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
            "common_abc": {}
        }
        # 1. Individual and combined FP
        fp_params = pd.read_csv(FP_FIT_FILEPATH, index_col=0)
        for survey in SURVEY_LIST:
            fp_methods_dict["individual"][survey] = fp_params.loc[survey][fp_base_params].to_numpy()
            fp_methods_dict["combined"][survey] = fp_params.loc["ALL_COMBINED"][fp_base_params].to_numpy()

        # 2. abc-fixed FP
        fp_params = pd.read_csv(FP_FIT_ABC_FIXED_FILEPATH, index_col=0)
        for survey in SURVEY_LIST:
            fp_methods_dict["common_abc"][survey] = fp_params.loc[survey][fp_base_params].to_numpy()

        # Iterate for all sample
        for survey in SURVEY_LIST:
            logger.info(f"Calculating log-distance ratio for {survey}")
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
            logdist_output_filepath = LOGDIST_OUTPUT_FILEPATH[survey]
            df.to_csv(logdist_output_filepath, index=False)

        logger.info('Fitting log-distance ratios successful!')
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.', exc_info=True)


if __name__ == "__main__":
    main()
