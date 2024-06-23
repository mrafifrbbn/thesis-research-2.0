import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from typing import List, Dict

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
COMPLETENESS_SETTING = int(os.environ.get('COMPLETENESS_SETTING'))
FP_SETTING = int(os.environ.get('FP_SETTING'))
# Add new data combinations here
NEW_SURVEY_LIST = SURVEY_LIST #(SURVEY_LIST + ['ALL_COMBINED']) if SMIN_SETTING == 1 else SURVEY_LIST

# Create logging instance
logger = get_logger('fit_logdist')

# Set the file paths
INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/all_combined.csv')
}

FP_FIT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_fits.csv')

LOGDIST_POSTERIOR_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/6dfgs_posterior.npy'),
    'SDSS': os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/sdss_posterior.npy'),
    'LAMOST': os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/lamost_posterior.npy'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/all_combined_posterior.npy')
}
create_parent_folder(LOGDIST_POSTERIOR_OUTPUT_FILEPATH)

LOGDIST_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/all_combined.csv')
}
create_parent_folder(LOGDIST_OUTPUT_FILEPATH)

CURVEFIT_COMPARISON_IMG_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/6dfgs.png'),
    'SDSS': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/sdss.png'),
    'LAMOST': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/lamost.png'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/p_setting_{FP_SETTING}/all_combined.png')
}
create_parent_folder(CURVEFIT_COMPARISON_IMG_FILEPATH)

POSTERIOR_SKEWNESS_IMG_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/6dfgs_skewness.png'),
    'SDSS': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/sdss_skewness.png'),
    'LAMOST': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/lamost_skewness.png'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'img/logdist/smin_setting_{SMIN_SETTING}/completeness_setting_{COMPLETENESS_SETTING}/fp_setting_{FP_SETTING}/all_combined_skewness.png')
}
create_parent_folder(POSTERIOR_SKEWNESS_IMG_FILEPATH)

def fit_logdist(
        survey: str,
        df: pd.DataFrame,
        smin: float,
        FPparams: List[float],
        mag_high: float = MAG_HIGH,
        mag_low: float = MAG_LOW,
        save_posterior: bool = False,
        logdist_posterior_filepath: str = None
    ) -> pd.DataFrame:
    """
    This is a function to calculate the log-distance ratio posteriors and obtain summary statistics.
    Summary statistics are obtained using two methods: direct calculation assuming skew-normal and 
    fitting using scipy's curve_fit assuming Gaussian.
    """
    # Get some redshift-distance lookup tables
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    dz = sp.interpolate.splev(df["z_cmb"].to_numpy()/LightSpeed, dist_spline)
    dz_cluster = sp.interpolate.splev(df["z_dist_est"], dist_spline)
    FPparams = np.array(FPparams)
    logger.info(f'Number of {survey} data = {len(df)}. FP best fits = {FPparams}')

    # Fit the logdistance ratios (range tested and number of points)
    dmin, dmax, nd = -1.5, 1.5, 1001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    # Calculate F_n (main part basically)
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    z_H = sp.interpolate.splev(d_H, red_spline, der=0)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    loglike = FP_func(FPparams, dbins, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), np.ones(len(df)), smin, sumgals=False)
    start = time.time()
    FNvals = FN_func(FPparams, df["z_cmb"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), lmin, lmax, smin)
    logger.info(f'Time elapsed to fit the logdists = {time.time() - start} s.')

    # Convert to the PDF for logdistance
    logP_dist = -1.5 * np.log(2.0 * math.pi) - loglike - FNvals

    # normalise logP_dist
    ddiff = np.log10(d_H[:-1]) - np.log10(d_H[1:])
    valdiff = np.exp(logP_dist[1:]) + np.exp(logP_dist[0:-1])
    norm_ = 0.5 * np.sum(valdiff * ddiff, axis=0)

    logP_dist -= np.log(norm_[:, None]).T

    # Calculate the mean and variance of the gaussian, then the skew
    mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0
    err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2)
    gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
    gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
    gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)
    delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
    scale = err * np.sqrt(1.0 / (1.0 - 2.0 * delta**2 / np.pi))
    loc = mean - scale * delta * np.sqrt(2.0 / np.pi)
    alpha = delta / (np.sqrt(1.0 - delta**2))

    # Store the skew-normal values to the dataframe
    df["logdist_mean"] = mean
    df["logdist_std"] = err
    df["logdist_alpha"] = alpha
    df["logdist_loc"] = loc
    df["logdist_scale"] = scale

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)
    # Save the posterior distribution
    if save_posterior:
        np.save(logdist_posterior_filepath, y)

    # Find mean and standard deviation of the distribution using curve_fit
    logdist_mean = []
    logdist_std = []
    chisq = []
    for i, y_ in enumerate(y):
        popt, pcov = curve_fit(gaus, dbins, y_, p0=[mean[i], err[i]])
        popt[1] = np.absolute(popt[1])
        logdist_mean.append(popt[0])
        logdist_std.append(popt[1])

        # Calculate the chi-squared statistic for the fit
        ypred = norm.pdf(dbins, popt[0], popt[1])
        chisquare = np.sum((y_ - ypred)**2 / ypred, axis=0)
        chisq.append(chisquare)

    df['logdist_mean_cf'] = logdist_mean
    df['logdist_std_cf'] = logdist_std
    df['logdist_chisq_cf'] = chisq
    
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

def main() -> None:
    try:
        logger.info(f"{'=' * 50}")
        logger.info('Fitting log-distance ratios...')
        logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}. FP_SETTING = {FP_SETTING}.')
        
        for survey in SURVEY_LIST:
            # Get input filename (outlier-rejected sample)
            input_filepath = INPUT_FILEPATH[survey]
            df = pd.read_csv(input_filepath)
            
            # Survey's veldisp limit
            if survey == 'ALL_COMBINED':
                smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING]['6dFGS']
            else:
                smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]
                
            # Get the FP parameters to be used for fitting
            FPparams = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc[SURVEY_FP_SETTING[FP_SETTING][survey]].to_numpy()
            
            # Logdist posterior filepath
            logdist_posterior_filepath = LOGDIST_POSTERIOR_OUTPUT_FILEPATH[survey]
            
            # Get output filename
            logdist_output_filepath = LOGDIST_OUTPUT_FILEPATH[survey]
            
            df = fit_logdist(
                survey=survey,
                df=df,
                smin=smin,
                FPparams=FPparams,
                logdist_output_filepath=logdist_output_filepath,
                save_posterior=True,
                logdist_posterior_filepath=logdist_posterior_filepath
                    )

            # Save the new dataframe
            df.to_csv(logdist_output_filepath, index=False)

        # logger.info('Comparing Gaussian curve_fit vs skew-normal...')
        # compare_cf_vs_formula()
        
        # logger.info('Plotting the best and worst posterior distributions...')
        # plot_best_worst_posterior()

        logger.info('Fitting log-distance ratios successful!')
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.')

if __name__ == '__main__':
    main()
