import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple
import emcee

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.utils.constants import *
from main_code.utils.functions import gaus
from main_code.utils.CosmoFunc import *
from main_code.utils.filepaths import (
    FOUNDATION_ZONE_FP_SAMPLE_FILEPATHS,
    FP_FIT_FILEPATH,
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
def FN_func(params, zobs, er, es, ei, lmin, lmax, smin):

    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params

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

# The likelihood function for the Fundamental Plane
def FP_func(var_params, const_params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals=True, chi_squared_only=False, use_full_fn=True):
    
    rmean, smean, imean, sigma1, sigma2, sigma3 = var_params
    a, c = const_params
    b = (rmean - a * smean - c) / imean

    params = np.array([a, b, rmean, smean, imean, sigma1, sigma2, sigma3])
    
    k = 0.0

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

    # Calculate full f_n
    if use_full_fn:
        FN = FN_func(params, z_obs, err_r, err_s, err_i, lmin, lmax, smin) + np.log(C_m)
    # Compute the FN term for the Scut only
    else:
        delta = (A*F**2 + I*B**2 - 2.0*B*C*F)/det
        FN = np.log(0.5 * special.erfc(np.sqrt(E/(2.0*(det+delta)))*(smin-smean)))/Sn + np.log(C_m)

    if chi_squared_only:
        return chi_squared
    elif sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0 * FN)
    else:
        return 0.5 * (chi_squared + log_det)


def fit_FP(
    survey: str,
    df: pd.DataFrame,
    smin: float,
    const_params: np.ndarray,
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
    
    print(f"{'=' * 10} Fitting {survey} Fundamental Plane | Ngals = {len(df)} {'=' * 10}")

    # Load peculiar velocity model
    print("Calculating predicted l.o.s. peculiar velocity from model")
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
    df['r_true'] = df['r'] - df['logdist_pred']

    # If using partial f_n, calculate Sn using the VMAX method
    if not USE_FULL_FN:
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
    # If using full f_n, set Sn=1 for all galaxies
    else:
        Sn = 1.0
        df['Sn'] = Sn

    # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
    data_fit = df
    badcount = len(df)
    is_converged = False
    i = 1
    
    while not is_converged:

        Snfit = data_fit["Sn"].to_numpy()

        # The range of the FP parameters' values
        rvals, svals, ivals = param_boundaries[0], param_boundaries[1], param_boundaries[2]
        s1vals, s2vals, s3vals = param_boundaries[3], param_boundaries[4], param_boundaries[5]

        # Fit the FP parameters
        FPparams = sp.optimize.differential_evolution(FP_func, bounds=(rvals, svals, ivals, s1vals, s2vals, s3vals), 
            args=(const_params, 0.0, data_fit["z_cmb"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy(), True, False, use_full_fn), maxiter=10000, tol=1.0e-6, workers=-1, seed=42)
        
        # Break from the loop if reject_outliers is set to false
        if reject_outliers == False:
            break
        
        # Calculate the chi-squared 
        chi_squared = Sn * FP_func(FPparams.x, const_params, 0.0, df["z_cmb"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, chi_squared_only=True)[0]
        
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
                      best_params: np.ndarray,
                      const_params: np.ndarray,
                      smin: float,
                      chain_output_filepath: str = None,
                      param_boundaries: List[Tuple] = PARAM_BOUNDARIES,
                      ) -> np.ndarray:
    # The log-prior function for the FP parameters
    def log_prior(theta):
        smean, imean = theta
        smean_bound, imean_bound, = param_boundaries
        if smean_bound[0] < smean < smean_bound[1] and imean_bound[0] < imean < imean_bound[1]:
            return 0.0
        else:
            return -np.inf

    # Calculate log-posterior distribution
    def log_probability(theta, const_params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals=True, chi_squared_only=False, use_full_fn=True):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - FP_func(theta, const_params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals, chi_squared_only, use_full_fn)
    
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
    pos = best_params + 1e-4 * np.random.randn(16, 2)
    nwalkers, ndim = pos.shape

    # Run the MCMC
    logger.info("Running the MCMC sampler")
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(const_params, 0.0, z, r, s, i, dr, ds, di, Sn, smin, lmin, lmax, C_m, True, False, USE_FULL_FN)
    )
    sampler.run_mcmc(pos, 1000, progress=True, skip_initial_state_check=True)

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


def fit_logdist(
        survey: str,
        df: pd.DataFrame,
        smin: float,
        FPlabel: str,
        var_params: np.ndarray,
        const_params: np.ndarray,
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
    print(f"Calculating the logdistance PDF for galaxies in {survey} | Var params: {list(var_params)} | Const params: {list(const_params)}")

    # Get some redshift-distance lookup tables
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    dz = sp.interpolate.splev(df["z_cmb"].to_numpy() / LIGHTSPEED, dist_spline)
    dz_cluster = sp.interpolate.splev(df["z_dist_est"], dist_spline)
    logger.info(f'Number of {survey} data = {len(df)}.')

    # Define the range of logdists values to be calculated
    dmin, dmax, nd = -1.5, 1.5, 2001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
    loglike = FP_func(var_params, const_params, dbins, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), np.ones(len(df)), smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, use_full_fn=use_full_fn)
    
    # Calculate full FN
    a, c = const_params
    rmean, smean, imean, sigma1, sigma2, sigma3 = var_params
    b = (rmean - a * smean - c) / imean
    params = np.array([a, b, rmean, smean, imean, sigma1, sigma2, sigma3])

    FNvals = FN_func(params, df["z_cmb"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), lmin, lmax, smin)

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

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)
    # Save the posterior distributions
    if save_posterior:
        logdist_posterior_filepath = os.path.join(ROOT_PATH, f'artifacts/logdist/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/{survey.lower()}_posterior_{FPlabel.lower()}_fp.npy')
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

    df['logdist'] = logdist_mean
    df['logdist_err'] = logdist_std
    df['logdist_chisq'] = chisq
    df['logdist_rmse'] = rmse

    # Calculate observational error
    logdist_int_err = sigma1 * np.sqrt(1 + a**2 + b**2)
    df["logdist_obs_err"] = np.sqrt(df["logdist_err"]**2 - logdist_int_err**2)
    
    return df

def main():
    # try:
    logger.info(f"{'=' * 50}")
    logger.info('Fitting log-distance ratios...')
    logger.info(f'Environment variable: SMIN_SETTING = {SMIN_SETTING}.')

    # Get the combined FP params
    filepath = "/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/artifacts/fp_fit/smin_setting_1/fp_fit_method_0/fp_fits_individual.csv"
    FPparams = pd.read_csv(filepath, index_col=0)
    FPparams = FPparams.loc["ALL_COMBINED"]
    a, b, sigma1, sigma2, sigma3 = FPparams[["a", "b", "s1", "s2", "s3"]]
    c = FPparams["rmean"] - a * FPparams["smean"] - b * FPparams["imean"]

    # Constant parameters
    const_params = np.array([a, c])
    print("Constant parameters: ", const_params)

    # Varied parameters: rmean, smean, imean, sigma1, sigma2, sigma3
    param_boundaries = [(-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.01, 0.06), (0.20, 0.45), (0.1, 0.25)]

    FP_params_bestfit = []
    for survey in SURVEY_LIST:
        # Get input filename (outlier-rejected sample)
        input_filepath = FOUNDATION_ZONE_FP_SAMPLE_FILEPATHS[survey]
        print("Input filepath: ", input_filepath)
        df = pd.read_csv(input_filepath)
        
        # Survey's veldisp limit
        smin = SURVEY_VELDISP_LIMIT[1][survey]

        print(f"Fitting the FP for {survey}. Treating rmean as dependent variable...")
        params, df_fitted = fit_FP(
            survey=survey,
            df=df,
            smin=smin,
            const_params=const_params,
            param_boundaries=param_boundaries,
            reject_outliers=True,
            use_full_fn=True,
            pvals_cut=PVALS_CUT,
        )

        # print(f"Likelihood samping for {survey}...")
        # logger.info(f"Sampling the FP likelihood for {survey}.")
        # chain_output_filepath = f"/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_024_vary_centroid/chains/{survey.lower()}_chain.npy"
        # params_mean = sample_likelihood(
        #     df=df_fitted,
        #     best_params=params,
        #     const_params=const_params,
        #     smin=smin,
        #     chain_output_filepath=chain_output_filepath,
        #     param_boundaries=param_boundaries
        #     )

        logger.info(f"Fitting logdist for {survey}. Treating rmean as dependent variable...")
        df = fit_logdist(
            survey=survey,
            df=df_fitted,
            smin=smin,
            FPlabel="rsi_varied",
            var_params=params.copy(),
            const_params=const_params,
            use_full_fn=True,
            save_posterior=False,
            )
        
        # # Store best-fit parameters
        # rmean = c + a * params[0] + b * params[1]
        # params = np.insert(params, [0], rmean)

        FP_params_bestfit.append(params.copy())
        print("FP_params_bestfit: ", FP_params_bestfit)

        # Save logdist measurements
        logdist_output_filepath = f"/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_025_fix_ac_only/b_dependent/{survey.lower()}.csv"
        df.to_csv(logdist_output_filepath, index=False)

    # Convert the FP parameters to dataframe and save to artifacts folder
    FP_params_bestfit = np.array(FP_params_bestfit)
    FP_columns = ['rmean', 'smean', 'imean', 's1', 's2', 's3']
    filepath_ = f"/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_025_fix_ac_only/b_dependent/fp_fits.csv"
    df = pd.DataFrame(FP_params_bestfit, columns=FP_columns, index=SURVEY_LIST)
    df["a"] = a
    df["b"] = (df["rmean"] - a * df["smean"] - c) / df["imean"]
    df["c"] = c
    df = df[["a", "b", "c", "rmean", "smean", "imean", "s1", "s2", "s3"]]
    df.to_csv(filepath_)


if __name__ == '__main__':
    main()
