import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.stats import norm

from scipy.odr import ODR, Model, RealData

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)
from main_code.utils.constants import *
from main_code.utils.functions import *
from main_code.utils.CosmoFunc import *
from main_code.utils.filepaths import H22_FILEPATH
from main_code.step_11_distance_modulus import calculate_distance_modulus

pvhub_dir = os.environ.get('PVHUB_DIR_PATH')
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

# Constants (modify before running)
SMIN = np.log10(70)
R_MAG_NEW_LIM = 15.28 # New magnitude limit (from notebook)
USE_FULL_FN = False
H0 = 74.6


def fit_FP(
    df: pd.DataFrame,
    smin: float = np.log10(70),
    param_boundaries: List[Tuple] = [(0.8, 1.5), (-1.0, -0.7), (-0.2, 0.4), (2.0, 2.5), (2.5, 3.0), (0.04, 0.1), (0.3, 0.5), (0.15, 0.25)],
    pvals_cut: float = 0.01,
    zmin: float = 0.0033,
    zmax: float = ZMAX,
    solar_magnitude: float = 4.65,
    mag_high: float = 17.0,
    mag_low: float = 10.0,
    reject_outliers: bool = False,
    use_full_fn: bool = False
          ) -> Tuple[np.ndarray, pd.DataFrame]:

    # Load peculiar velocity model
    pv_model = TwoMPP_SDSS_6dF(verbose=True) # type: ignore

    # Calculate predicted PVs using observed group redshift in CMB frame, and calculate cosmological redshift
    df['v_pec'] = pv_model.calculate_pv(df['ra'].to_numpy(), df['dec'].to_numpy(), df['z_dist_est'].to_numpy())
    df['z_pec'] = df['v_pec'] / LIGHTSPEED
    df['z_cosmo'] = ((1 + df['z_dist_est']) / (1 + df['z_pec'])) - 1
    
    # Calculate predicted true distance and FN integral limits
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    d_H = sp.interpolate.splev(df['z_cosmo'].to_numpy(), dist_spline, der=0)
    df['lmin'] = (solar_magnitude + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_r"].to_numpy() + df["extinction_r"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    df['lmax'] = (solar_magnitude + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_r"].to_numpy() + df["extinction_r"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0

    # Calculate predicted logdistance-ratios
    d_z = sp.interpolate.splev(df['z_dist_est'].to_numpy(), dist_spline, der=0)
    df['logdist_pred'] = np.log10(d_z / d_H)
    df['r_true'] = df['r']# - df['logdist_pred']

    # If using partial f_n, calculate Sn using the VMAX method
    if not use_full_fn:
        # Get some redshift-distance lookup tables
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        # The comoving distance to each galaxy using group redshift as distance indicator
        dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0)

        # (1+z) factor because we use luminosity distance
        Vmin = (1.0 + zmin)**3 * sp.interpolate.splev(zmin, dist_spline)**3
        Vmax = (1.0 + zmax)**3 * sp.interpolate.splev(zmax, dist_spline)**3
        # Maximum (luminosity) distance the galaxy can be observed given MAG_HIGH (survey limiting magnitude)
        Dlim = 10.0**((mag_high - (df["deVMag_r"] - df['extinction_r']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)    
        # Find the corresponding maximum redshift
        zlim = sp.interpolate.splev(Dlim, lumred_spline)
        Sn = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))
        df['Sn'] = Sn
    # If using full f_n, set Sn=1 for all galaxies
    else:
        print("Assuming Sn = 1")
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
        chi_squared = Sn * FP_func(FPparams.x, 0.0, df["z_cmb"].to_numpy(), df["r_true"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), df["C_m"].to_numpy(), sumgals=False, chi_squared_only=True)[0]
        
        # Calculate the p-value (x,dof)
        pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
        
        # Reject galaxies with p-values < pvals_cut (probabilities of being part of the sample lower than some threshold)
        data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
        
        # Count the number of rejected galaxies
        badcountnew = len(np.where(pvals < pvals_cut)[0])
        
        # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
        is_converged = True if badcount == badcountnew else False
        
        # Set the new count of rejected galaxies
        badcount = badcountnew
        i += 1

    df = data_fit
    
    return FPparams.x, df


def fit_logdist(
        df: pd.DataFrame,
        smin: float,
        FPparams: np.ndarray,
        use_full_fn: bool = True,
        mag_high: float = MAG_HIGH,
        mag_low: float = MAG_LOW
    ) -> pd.DataFrame:
    """
    This is a function to calculate the log-distance ratio posteriors and obtain summary statistics.
    Summary statistics are obtained using two methods: direct calculation assuming skew-normal and 
    fitting using scipy's curve_fit assuming Gaussian.
    """

    # Get some redshift-distance lookup tables
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    dz = sp.interpolate.splev(df["z_cmb"].to_numpy() / LIGHTSPEED, dist_spline)
    dz_cluster = sp.interpolate.splev(df["z_dist_est"], dist_spline)

    # Define the range of logdists values to be calculated
    dmin, dmax, nd = -1.5, 1.5, 2001
    dbins = np.linspace(dmin, dmax, nd, endpoint=True)

    # Calculate full FN
    d_H = np.outer(10.0**(-dbins), dz_cluster)
    lmin = (4.65 + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_r"].to_numpy() + df["extinction_r"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_high) / 5.0
    lmax = (4.65 + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_r"].to_numpy() + df["extinction_r"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - mag_low) / 5.0
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

    # Transpose the PDF and return to linear unit
    y = np.exp(logP_dist.T)

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

    df[f'logdist_new'] = logdist_mean
    df[f'logdist_err_new'] = logdist_std
    
    return df


def main():
    col_mapping = {
        "RA": "ra",
        "Dec": "dec",
        "zcmb": "z_cmb",
        "zcmb_group": "z_dist_est",
    }

    # Load cullan's raw data and rename the columns
    df = pd.read_csv(H22_FILEPATH, delim_whitespace=True)
    df['objid'] = df['objid'].apply(lambda x: 'SDSS' + str(x))
    df.rename(col_mapping, axis=1, inplace=True)
    df["C_m"] = 1.0
    print(f"Number of data (before): {len(df)}")

    # Filter data: redshift, magnitude, veldisp
    df = df[(df['z_dist_est'] >= ZMIN) & (df['z_dist_est'] <= ZMAX)]
    df = df[(df['deVMag_r'] - df['extinction_r']) <= R_MAG_NEW_LIM]
    df = df[df['s'] >= SMIN]
    print(f"Number of data (after): {len(df)}")

    # Fit FP
    fp_params, df_clean = fit_FP(
        df=df,
        smin=SMIN,
        param_boundaries = [(0.8, 1.5), (-1.0, -0.7), (-0.2, 0.4), (2.0, 2.5), (2.5, 3.0), (0.04, 0.1), (0.2, 0.5), (0.15, 0.25)],
        pvals_cut=0.01,
        zmin=ZMIN,
        zmax=ZMAX,
        solar_magnitude=4.65,
        mag_high=R_MAG_NEW_LIM,
        mag_low=MAG_LOW,
        reject_outliers=True,
        use_full_fn=USE_FULL_FN
    )
    print(f"Remaining data: {len(df_clean)}")
    print(f"Best-fit params: {fp_params}")

    # Fit logdist
    df_logdist = fit_logdist(
        df=df_clean,
        smin=SMIN,
        FPparams=fp_params,
        use_full_fn=USE_FULL_FN,
        mag_high=R_MAG_NEW_LIM,
        mag_low=MAG_LOW
    )

    # Calculate distance moduli
    df_logdist["DM_old"], df_logdist["eDM_old"] = calculate_distance_modulus(df_logdist["z_dist_est"].to_numpy(), df_logdist["zhelio"].to_numpy(), df_logdist[f"logdist_corr"].to_numpy(), df_logdist[f"logdist_corr_err"].to_numpy(), H0=H0)

    df_logdist["DM_new"], df_logdist["eDM_new"] = calculate_distance_modulus(df_logdist["z_dist_est"].to_numpy(), df_logdist["zhelio"].to_numpy(), df_logdist[f"logdist_new"].to_numpy(), df_logdist[f"logdist_err_new"].to_numpy(), H0=H0)

    # Save data
    if USE_FULL_FN:
        filename = f"cullan_sdss_logdist_refit_full_fn.csv"
    else:
        filename = f"cullan_sdss_logdist_refit_partial_fn.csv"
    filepath = os.path.join(ROOT_PATH, f"experiments/experiment_011_refit_cullan_sdss/{filename}")
    df_logdist.to_csv(filepath, index=False)


if __name__ == '__main__':
    main()