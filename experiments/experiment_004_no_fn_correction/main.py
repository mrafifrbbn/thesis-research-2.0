import os
import sys
import trace
import math
import numpy as np
import pandas as pd
import scipy as sp
import traceback
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib import cm, ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv(override=True)

root_dir = os.environ.get("ROOT_PATH")
if not root_dir in sys.path: sys.path.append(root_dir)

from experiments.experiment_004_no_fn_correction.utils import rz_table, FP_func
from src.utils.constants import *
from src.utils.logging_config import get_logger

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

import emcee

# Get environment variables from .env file
ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
COMPLETENESS_SETTING = int(os.environ.get('COMPLETENESS_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Add new data combinations here
NEW_SURVEY_LIST = SURVEY_LIST + ['ALL_COMBINED'] if SMIN_SETTING == 1 else SURVEY_LIST

# Create logging instance
logger = get_logger('fit_fp')

# Set global random seed
np.random.seed(42)

# CONSTANTS #
USE_FULL_FN = False


def fit_FP(
    survey: str,
    df: pd.DataFrame,
    smin: float,
    param_boundaries: List[Tuple] = PARAM_BOUNDARIES,
    pvals_cut: float = PVALS_CUT,
    zmin: float = ZMIN,
    zmax: float = ZMAX,
    solar_magnitude: float = SOLAR_MAGNITUDE.get('j'),
    mag_high: float = MAG_HIGH,
    mag_low: float = MAG_LOW,
    reject_outliers: bool = REJECT_OUTLIERS,
    use_full_fn: bool = True
          ) -> Tuple[np.ndarray, pd.DataFrame]:
    
    logger.info(f"{'=' * 10} Fitting {survey} Fundamental Plane | Ngals = {len(df)} {'=' * 10}")

    # Load peculiar velocity model
    logger.info("Calculating predicted l.o.s. peculiar velocity from model")
    pv_model = TwoMPP_SDSS_6dF(verbose=True) # type: ignore

    # Calculate predicted PVs using observed group redshift in CMB frame, and calculate cosmological redshift
    df['v_pec'] = pv_model.calculate_pv(df['ra'].to_numpy(), df['dec'].to_numpy(), df['z_dist_est'].to_numpy())
    df['z_pec'] = df['v_pec'] / LIGHTSPEED
    df['z_cosmo'] = ((1 + df['z_dist_est']) / (1 + df['z_pec'])) - 1
    
    # Calculate predicted true distance and FN integral limits
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    d_H = sp.interpolate.splev(df['z_cosmo'].to_numpy(), dist_spline, der=0)
    df['lmin'] = (solar_magnitude + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    df['lmax'] = (solar_magnitude + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0

    # Calculate predicted logdistance-ratios
    d_z = sp.interpolate.splev(df['z_dist_est'].to_numpy(), dist_spline, der=0)
    df['logdist_pred'] = np.log10(d_z / d_H)
    df['r_true'] = df['r'] # - df['logdist_pred']

    # # Set Sn = 1.0
    # Sn = 1.0
    # df['Sn'] = Sn

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


    # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
    data_fit = df
    badcount = len(df)
    is_converged = False
    i = 1
    
    while not is_converged:

        Snfit = data_fit["Sn"].to_numpy()

        # The range of the FP parameters' values
        avals, bvals = param_boundaries[0], param_boundaries[1]
        rvals, svals, ivals = param_boundaries[2], param_boundaries[3], param_boundaries[4]
        s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

        # Fit the FP parameters
        FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
            args=(0.0, data_fit["z_cmb"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, use_full_fn), maxiter=10000, tol=1.0e-6, workers=-1, seed=42)
        
        # Break from the loop if reject_outliers is set to false
        if reject_outliers == False:
            break
        
        # Calculate the chi-squared 
        chi_squared = Sn * FP_func(FPparams.x, 0.0, df["z_cmb"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, chi_squared_only=True, use_full_fn=USE_FULL_FN)[0]
        
        # Calculate the p-value (x,dof)
        pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
        
        # Reject galaxies with p-values < pvals_cut (probabilities of being part of the sample lower than some threshold)
        data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
        
        # Count the number of rejected galaxies
        badcountnew = len(np.where(pvals < pvals_cut)[0])
        
        # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
        is_converged = True if badcount == badcountnew else False

        # logger.info verbose
        logger.info(f"{'-' * 10} Iteration {i} {'-' * 10} | FP parameters: {FPparams.x.tolist()}")
        
        # Set the new count of rejected galaxies
        badcount = badcountnew
        i += 1

    df = data_fit
    logger.info(f'Number of galaxies remaining: {len(df)}')
    
    return FPparams.x, df

def sample_likelihood(df: pd.DataFrame,
                      FP_params: np.ndarray, 
                      smin: float,
                      chain_output_filepath: str = None,
                      use_full_fn: bool = True,
                      param_boundaries: List[Tuple] = PARAM_BOUNDARIES
                      ) -> np.ndarray:
    # The log-prior function for the FP parameters
    def log_prior(theta):
        a, b, rmean, smean, imean, sig1, sig2, sig3 = theta
        a_bound, b_bound, rmean_bound, smean_bound, imean_bound, s1_bound, s2_bound, s3_bound = param_boundaries
        if a_bound[0] < a < a_bound[1] and b_bound[0] < b < b_bound[1] and rmean_bound[0] < rmean < rmean_bound[1] and smean_bound[0] < smean < smean_bound[1] and imean_bound[0] < imean < imean_bound[1] and s1_bound[0] < sig1 < s1_bound[1] and s2_bound[0] < sig2 < s2_bound[1] and s3_bound[0] < sig3 < s3_bound[1]:
            return 0.0
        else:
            return -np.inf

    # Calculate log-posterior distribution
    def log_probability(theta, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals, chi_squared_only, use_full_fn):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - FP_func(theta, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals, chi_squared_only, use_full_fn)
    
    # Load the observables needed to sample the likelihood
    z = df['z_cmb'].to_numpy()
    r = df['r_true'].to_numpy()
    s = df['s'].to_numpy()
    i = df['i'].to_numpy()
    dr = df['er'].to_numpy()
    ds = df['es'].to_numpy()
    di = df['ei'].to_numpy()
    lmin = df['lmin'].to_numpy()
    lmax = df['lmax'].to_numpy()
    C_m = df['C_m'].to_numpy()
    Sn = df['Sn'].to_numpy()

    # Specify the initial guess, the number of walkers, and dimensions
    pos = FP_params + 1e-4 * np.random.randn(16, 8)
    nwalkers, ndim = pos.shape

    # Run the MCMC
    logger.info("Running the MCMC sampler")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(0.0, z, r, s, i, dr, ds, di, Sn, smin, lmin, lmax, C_m, True, False, use_full_fn)
    )
    sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True)

    # Flatten the chain and save as numpy array
    logger.info("Flattening the chain and saving them as numpy array")
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    if chain_output_filepath is not None:
        np.save(chain_output_filepath, flat_samples)

    # Get the mean values of the marginalized distribution
    logger.info("Get the mean values of the marginalized distributions.")
    x_ = flat_samples.T
    FP_params_mean = np.mean(x_, axis=1)

    return FP_params_mean


def main():
    try:
        logger.info("Running experiment 004: fitting the FP without Malmquist bias correction for the combined sample")

        # Load the data
        input_filepath = os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_1/all_combined.csv')
        df = pd.read_csv(input_filepath)

        # Set 6dFGS veldisp lower limit
        smin = SURVEY_VELDISP_LIMIT[1]['6dFGS']

        # Custom parameter boundaries
        param_boundaries = [(1.2, 2.5), (-1.1, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.0, 0.06), (0.20, 0.45), (0.1, 0.25)]

        params, df_fitted = fit_FP(
                survey="ALL_COMBINED",
                df=df,
                smin=smin,
                param_boundaries=param_boundaries,
                reject_outliers=True,
                use_full_fn=USE_FULL_FN,
                pvals_cut=PVALS_CUT
            )
        FP_params = np.array([params])
        FP_columns = ['a', 'b', 'rmean', 'smean', 'imean', 's1', 's2', 's3']
        pd.DataFrame(FP_params, columns=FP_columns, index=["ALL_COMBINED"]).to_csv("fp_fits.csv")

        # Save the cleaned sample
        output_filepath = os.path.join(ROOT_PATH, "outlier_rejects.csv")
        df_fitted.to_csv(output_filepath, index=False)

        # Sample the likelihood
        chain_output_filepath = os.path.join(ROOT_PATH, "experiments/experiment_004_no_fn_correction/mcmc_chain.npy")
        params_mean = sample_likelihood(
            df=df_fitted,
            FP_params=params,
            smin=smin,
            chain_output_filepath=chain_output_filepath,
            use_full_fn=USE_FULL_FN,
            param_boundaries=param_boundaries
        )
        
    except Exception as e:
        logger.error(f'Fitting the FP failed. Reason: {e}.')
        logger.error(f"Traceback: {traceback.print_exc()}")


if __name__ == '__main__':
    main()