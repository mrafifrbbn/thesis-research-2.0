import os
import sys
import time
import numpy as np
import pandas as pd
import math
import scipy as sp
from scipy import special
from typing import List, Tuple
import emcee

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.utils.constants import *
from main_code.utils.functions import create_parent_folder
from main_code.utils.CosmoFunc import rz_table
from main_code.utils.filepaths import (
    OUTLIER_REJECT_FP_SAMPLE_FILEPATHS,
    FP_FIT_FILEPATH,
    MCMC_CHAIN_ABC_FIXED_FILEPATH,
)
from main_code.utils.logging_config import get_logger

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Create logging instance
logger = get_logger('fit_logdist')


# Calculates f_n (the integral over the censored 3D Gaussian of the Fundamental Plane) for a magnitude limit and velocity dispersion cut. 
def FN_func(FPparams, zobs, er, es, ei, lmin, lmax, smin):

    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = FPparams
    k = 0.0

    fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
    norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
    dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
    sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
    sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
    sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
    sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
    sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
    sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2

    err_r = er**2 + np.log10(1.0 + 300.0/(LIGHTSPEED*zobs))**2 + sigmar2
    err_s = es**2 + sigmas2
    err_i = ei**2 + sigmai2
    cov_ri = -1.0*er*ei + sigmari

    A = err_s*err_i - sigmasi**2
    B = sigmasi*cov_ri - sigmars*err_i
    C = sigmars*sigmasi - err_s*cov_ri
    E = err_r*err_i - cov_ri**2
    F = sigmars*cov_ri - err_r*sigmasi
    I = err_r*err_s - sigmars**2

    # Inverse of the determinant!!
    det = 1.0/(err_r*A + sigmars*B + cov_ri*C)

    # Compute all the G, H and R terms
    G = np.sqrt(E)/(2*F-B)*(C*(2*F+B) - A*F - 2.0*B*I)
    delta = (I*B**2 + A*F**2 - 2.0*B*C*F)*det**2
    Edet = E*det
    Gdet = (G*det)**2
    Rmin = (lmin - rmean - imean/2.0)*np.sqrt(2.0*delta/det)/(2.0*F-B)
    Rmax = (lmax - rmean - imean/2.0)*np.sqrt(2.0*delta/det)/(2.0*F-B)

    G0 = -np.sqrt(2.0/(1.0+Gdet))*Rmax
    G2 = -np.sqrt(2.0/(1.0+Gdet))*Rmin
    G1 = -np.sqrt(Edet/(1.0+delta))*(smin - smean)

    H = np.sqrt(1.0+Gdet+delta)
    H0 = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmax
    H2 = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmin
    H1 = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmax/(smin - smean)
    H3 = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmin/(smin - smean)

    FN = special.owens_t(G0, H0/H)+special.owens_t(G1, H1/H)-special.owens_t(G2, H2/H)-special.owens_t(G1, H3/H)
    FN += 1.0/(2.0*np.pi)*(np.arctan2(H2,H)+np.arctan2(H3,H)-np.arctan2(H0,H)-np.arctan2(H1,H))
    FN += 1.0/4.0*(special.erf(G0/np.sqrt(2.0))-special.erf(G2/np.sqrt(2.0)))

    # This can go less than zero for very large distances if there are rounding errors, so set a floor
    # This shouldn't affect the measured logdistance ratios as these distances were already very low probability!
    index = np.where(FN < 1.0e-15)
    FN[index] = 1.0e-15

    return np.log(FN)

# Likelihood for a single data set and a set of FP parameters
def FP_func(fp_params, df, smin, sumgals=True, chi_squared_only=False):

    # Unpack FP parameters
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = fp_params

    k = 0.0

    # Unpack FP observables
    logdists = 0.0
    z_obs = df["z_cmb"].to_numpy()
    r = df["r_true"].to_numpy()
    s = df["s"].to_numpy()
    i = df["i"].to_numpy()
    err_r = df["er"].to_numpy()
    err_s = df["es"].to_numpy()
    err_i = df["ei"].to_numpy()
    Sn = df["Sn"].to_numpy()
    lmin = df["lmin"].to_numpy()
    lmax = df["lmax"].to_numpy()

    fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
    norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
    dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
    sigmar2 =  1.0/norm1*sigma1**2 + b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
    sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
    sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
    sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
    sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
    sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2

    # Compute the chi-squared and determinant (quickly!)
    cov_r = err_r**2 + np.log10(1.0 + 300.0/(LIGHTSPEED*z_obs))**2 + sigmar2
    cov_s = err_s**2 + sigmas2
    cov_i = err_i**2 + sigmai2
    cov_ri = -1.0*err_r*err_i + sigmari

    A = cov_s*cov_i - sigmasi**2
    B = sigmasi*cov_ri - sigmars*cov_i
    C = sigmars*sigmasi - cov_s*cov_ri
    E = cov_r*cov_i - cov_ri**2
    F = sigmars*cov_ri - cov_r*sigmasi
    I = cov_r*cov_s - sigmars**2	

    sdiff, idiff = s - smean, i - imean
    rnew = r - np.tile(logdists, (len(r), 1)).T
    rdiff = rnew - rmean

    det = cov_r*A + sigmars*B + cov_ri*C
    log_det = np.log(det)/Sn

    chi_squared = (A*rdiff**2 + E*sdiff**2 + I*idiff**2 + 2.0*rdiff*(B*sdiff + C*idiff) + 2.0*F*sdiff*idiff)/(det*Sn)

    # Calculate f_n correction
    FN = FN_func(fp_params, z_obs, err_r, err_s, err_i, lmin, lmax, smin)

    if chi_squared_only:
        return chi_squared
    elif sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0 * FN)
    else:
        return 0.5 * (chi_squared + log_det)


# Likelihood function to be maximized (sum of the likelihood of each survey)
def likelihood_function(params, df_6df, df_sdss, df_lamost, smin):

    # The common parameters: a, b, and c
    a, b, c = params[0], params[1], params[2]
    sigma1, sigma2, sigma3 = params[3], params[4], params[5]

    # Each survey's FP
    smean_6df, imean_6df = params[6], params[7]
    smean_sdss, imean_sdss = params[8], params[9]
    smean_lamost, imean_lamost = params[10], params[11]

    # Calculate rmean
    rmean_6df = c + a * smean_6df + b * imean_6df
    rmean_sdss = c + a * smean_sdss + b * imean_sdss
    rmean_lamost = c + a * smean_lamost + b * imean_lamost

    # Create arrays of FP parameters
    fp_params_6df = np.array([a, b, rmean_6df, smean_6df, imean_6df, sigma1, sigma2, sigma3])
    fp_params_sdss = np.array([a, b, rmean_sdss, smean_sdss, imean_sdss, sigma1, sigma2, sigma3])
    fp_params_lamost = np.array([a, b, rmean_lamost, smean_lamost, imean_lamost, sigma1, sigma2, sigma3])

    # Calculate (negative of) the likelihood for each survey
    likelihood_6df = FP_func(fp_params_6df, df_6df, smin, True, False)
    likelihood_sdss = FP_func(fp_params_sdss, df_sdss, smin, True, False)
    likelihood_lamost = FP_func(fp_params_lamost, df_lamost, smin, True, False)

    return likelihood_6df + likelihood_sdss + likelihood_lamost


def sample_likelihood(dfs: List[pd.DataFrame],
                      smin: float,
                      chain_output_filepath: str,
                      param_boundaries: List[Tuple] = PARAM_BOUNDARIES
                      ) -> np.ndarray:
    
    # The log-prior function for the FP parameters
    def log_prior(theta):

        # Unpack parameters
        a, b, c = theta[0], theta[1], theta[2]
        sigma1, sigma2, sigma3 = theta[3], theta[4], theta[5]
        smean_6df, imean_6df = theta[6], theta[7]
        smean_sdss, imean_sdss = theta[8], theta[9]
        smean_lamost, imean_lamost = theta[10], theta[11]

        # The range of the FP parameters' values
        avals, bvals, cvals = param_boundaries[0], param_boundaries[1], param_boundaries[2]
        svals, ivals = param_boundaries[3], param_boundaries[4]
        s1vals, s2vals, s3vals = param_boundaries[5], param_boundaries[6], param_boundaries[7]

        if (avals[0] < a < avals[1]) and (bvals[0] < b < bvals[1]) and (cvals[0] < c < cvals[1]) \
            and (s1vals[0] < sigma1 < s1vals[1]) and (s2vals[0] < sigma2 < s2vals[1]) and (s3vals[0] < sigma3 < s3vals[1]) \
            and (svals[0] < smean_6df < svals[1]) and (ivals[0] < imean_6df < ivals[1]) \
            and (svals[0] < smean_sdss < svals[1]) and (ivals[0] < imean_sdss < ivals[1]) \
            and (svals[0] < smean_lamost < svals[1]) and (ivals[0] < imean_lamost < ivals[1]):
            return 0.0
        else:
            return -np.inf

    # Calculate log-posterior distribution
    def log_probability(theta, dfs, smin):

        # Unpack data
        df_6df, df_sdss, df_lamost = dfs

        # Calculate prior
        lp = log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - likelihood_function(theta, df_6df, df_sdss, df_lamost, smin)

    # Load individual best-fit FP as initial guesses
    filepath = FP_FIT_FILEPATH
    fp_individual = pd.read_csv(filepath, index_col=0)

    # Unpack initial guesses
    a, b, c, sigma1, sigma2, sigma3 = fp_individual.loc["ALL_COMBINED"][["a", "b", "c", "s1", "s2", "s3"]]
    smean_6df, imean_6df = fp_individual.loc["6dFGS"][["smean", "imean"]]
    smean_sdss, imean_sdss = fp_individual.loc["SDSS"][["smean", "imean"]]
    smean_lamost, imean_lamost = fp_individual.loc["LAMOST"][["smean", "imean"]]

    # Pack initial guesses as an array
    FP_params = np.array([
        a, b, c, sigma1, sigma2, sigma3,
        smean_6df, imean_6df, 
        smean_sdss, imean_sdss, 
        smean_lamost, imean_lamost
        ])

    # Specify the initial guess, the number of walkers, and dimensions
    pos = FP_params + 1e-4 * np.random.randn(2 * len(FP_params), len(FP_params))
    nwalkers, ndim = pos.shape

    # Run the MCMC
    logger.info("Running the MCMC sampler")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(dfs, smin)
    )
    sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True)

    # Flatten the chain and save as numpy array
    logger.info("Flattening the chain and saving them as numpy array")
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    # Save MCMC chain
    np.save(chain_output_filepath, flat_samples)

    # Get the mean values of the marginalized distribution
    logger.info("Get the mean values of the marginalized distributions.")
    x_ = flat_samples.T
    FP_params_mean = np.mean(x_, axis=1)

    return FP_params_mean


def main():
    try:
        logger.info(f"{'=' * 50}")
        logger.info('Fitting log-distance ratios ')
        logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}.')

        # Set parameter boundaries
        param_boundaries = [(1.2, 2.0), (-1.0, -0.7), (-0.6, 0.2), (2.1, 2.4), (3.1, 3.5), (0.03, 0.06), (0.21, 0.35), (0.1, 0.2)]

        # Load peculiar velocity model
        print("Loading 2M++ SDSS-6dFGS PV model...")
        pv_model = TwoMPP_SDSS_6dF(verbose=True) # type: ignore

        # Survey's veldisp limit
        smin = SURVEY_VELDISP_LIMIT[1]["6dFGS"]

        dfs = []    # List of dataframes
        for survey in SURVEY_LIST:

            # Get data (outlier-rejected sample)
            input_filepath = OUTLIER_REJECT_FP_SAMPLE_FILEPATHS[survey]
            logger.info(f"Input filepath: {input_filepath}")
            df = pd.read_csv(input_filepath)

            # Calculate predicted PVs using observed group redshift in CMB frame, and calculate cosmological redshift
            df["v_pec"] = pv_model.calculate_pv(df["ra"].to_numpy(), df["dec"].to_numpy(), df["z_dist_est"].to_numpy())
            df["z_pec"] = df["v_pec"] / LIGHTSPEED
            df["z_cosmo"] = ((1 + df["z_dist_est"]) / (1 + df["z_pec"])) - 1
            
            # Calculate predicted true distance and FN integral limits
            red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
            d_H = sp.interpolate.splev(df['z_cosmo'].to_numpy(), dist_spline, der=0)
            df["lmin"] = (SOLAR_MAGNITUDE["j"] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_HIGH) / 5.0
            df["lmax"] = (SOLAR_MAGNITUDE["j"] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0

            # Calculate predicted logdistance-ratios
            d_z = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0)
            df["logdist_pred"] = np.log10(d_z / d_H)
            df["r_true"] = df["r"] - df["logdist_pred"]

            # Add constant Sn
            df["Sn"] = 1.0

            dfs.append(df)

        logger.info(f"Sampling 18-parameters FP likelihood.")
        chain_output_filepath = "/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_027_fp_fits_other_models/vary_centroid_only/chain.npy"
        create_parent_folder(chain_output_filepath)
        params_mean = sample_likelihood(
            dfs=dfs,
            smin=smin,
            chain_output_filepath=chain_output_filepath,
            param_boundaries=param_boundaries
            )

        logger.info('Fitting log-distance ratios successful!')
    
    except Exception as e:
        logger.error(f'Fitting log-distance ratios failed. Reason: {e}.', exc_info=True)


if __name__ == '__main__':
    main()