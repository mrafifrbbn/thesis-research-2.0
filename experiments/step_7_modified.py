import os
import sys
import numpy as np
import pandas as pd

import scipy as sp
from scipy import integrate, interpolate, special
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf

from matplotlib import cm, ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import List, Dict

src_dir = '/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/src'
if not src_dir in sys.path: sys.path.append(src_dir)
utils_dir = '/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/src/utils'
if not utils_dir in sys.path: sys.path.append(utils_dir)
pvhub_dir = '/Users/mrafifrbbn/Documents/thesis/pvhub/'
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger
from pvhub import *

import emcee
import getdist
from getdist import plots, MCSamples

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_SETTING = int(os.environ.get('FP_SETTING'))
COMPLETENESS_SETTING = int(os.environ.get('COMPLETENESS_SETTING'))

USE_FULL_FN = True

# FN_function
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

# Modified FP function
# The likelihood function for the Fundamental Plane
def modified_FP_func(params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, sumgals=True, chi_squared_only=False):

    if USE_FULL_FN:
        Sn = 1.0

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

    sigma_cov = np.array([[sigmar2, sigmars, sigmari], [sigmars, sigmas2, sigmasi], [sigmari, sigmasi, sigmai2]])

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

    # Compute the full FN
    if USE_FULL_FN:
        FN = FN_func(params, z_obs, err_r, err_s, err_i, lmin, lmax, smin)
    else:
        # Compute the FN term for the Scut only
        delta = (A*F**2 + I*B**2 - 2.0*B*C*F)/det
        FN = np.log(0.5 * special.erfc(np.sqrt(E/(2.0*(det+delta)))*(smin-smean)))/Sn
        

    if chi_squared_only:
        return chi_squared
    elif sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0 * FN)
    else:
        return 0.5 * (chi_squared + log_det)

if __name__ == "__main__":

    # Load peculiar velocity model
    pv_model = TwoMPP_SDSS_6dF(verbose=True)

    # Load the 6dFGS data
    data_filepath = os.path.join(ROOT_PATH, 'data/foundation/fp_sample/smin_setting_1/6dfgs.csv')
    df = pd.read_csv(data_filepath)

    # Calculate predicted PVs (using observed group redshift in CMB frame) and redshift
    redshift_col = 'z_cmb' # ONLY FOR THIS BLOCK
    df['v_pec'] = pv_model.calculate_pv(df['ra'].to_numpy(), df['dec'].to_numpy(), df[redshift_col].to_numpy())
    df['z_pec'] = df['v_pec'] / LIGHTSPEED
    df['z_cosmo'] = ((1 + df[redshift_col]) / (1 + df['z_pec'])) - 1

    # Calculate integral limits
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
    d_H = sp.interpolate.splev(df["z_cosmo"].to_numpy(), dist_spline, der=0)
    lmin = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_HIGH) / 5.0
    lmax = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0
    df['lmin'] = lmin
    df['lmax'] = lmax

    # Calculate Sn (VMAX method)
    dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0)
    Vmin = (1.0 + ZMIN)**3 * sp.interpolate.splev(ZMIN, dist_spline)**3
    Vmax = (1.0 + ZMAX)**3 * sp.interpolate.splev(ZMAX, dist_spline)**3
    Dlim = 10.0**((MAG_HIGH - (df["j_m_ext"] - df['extinction_j']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)
    zlim = sp.interpolate.splev(Dlim, lumred_spline)
    Sn = np.where(zlim >= ZMAX, 1.0, np.where(zlim <= ZMIN, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))
    df['Sn'] = Sn
    if USE_FULL_FN:
        df['Sn'] = 1.0

    # Some other constants
    smin = SURVEY_VELDISP_LIMIT[1]['6dFGS']
    pvals_cut = 0.01
    reject_outliers = True

    # To store the parameters
    params_list = []
    for i in range(1):
        # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
        data_fit = df.copy()
        badcount = len(df)
        is_converged = False
        i = 1

        while not is_converged:
            dz_cluster_fit = sp.interpolate.splev(data_fit['z_dist_est'].to_numpy(), dist_spline)
            Dlim = 10.0**((MAG_HIGH - (data_fit["j_m_ext"] - data_fit['extinction_j']).to_numpy() + 5.0 * np.log10(dz_cluster_fit) + 5.0 * np.log10(1.0 + data_fit["zhelio"])) / 5.0)
            zlim = sp.interpolate.splev(Dlim, lumred_spline)
            Snfit = np.where(zlim >= ZMAX, 1.0, np.where(zlim <= ZMIN, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))
            if USE_FULL_FN:
                Snfit = 1.0
            print("2, ", Snfit)

            # The range of the FP parameters' values
            avals, bvals = (1.0, 1.8), (-1.0, -0.5)
            rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.5), (3.0, 3.5)
            s1vals, s2vals, s3vals = (0., 0.3), (0.1, 0.5), (0.05, 0.3)

            # Fit the FP parameters
            if USE_FULL_FN:
                redshift_col = 'z_cosmo'
            else:
                redshift_col = 'z_cmb'
            print("3, ", redshift_col)
            FPparams = sp.optimize.differential_evolution(modified_FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit[redshift_col].to_numpy(), data_fit["r"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy()), maxiter=10000, tol=1.0e-6, workers=-1)
            # Calculate the chi-squared 
            chi_squared = Sn * modified_FP_func(FPparams.x, 0.0, df[redshift_col].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, smin, df["lmin"].to_numpy(), df["lmax"].to_numpy(), sumgals=False, chi_squared_only=True)[0]

            # Calculate the p-value (x,dof)
            pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
            # Reject galaxies with p-values < pvals_cut
            data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
            # Count the number of rejected galaxies
            badcountnew = len(np.where(pvals < pvals_cut)[0])
            # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
            is_converged = True if badcount == badcountnew else False

            # print verbose
            print(f"{'-' * 10} Iteration {i} {'-' * 10}")
            print(FPparams.x.tolist())
            a_fit, b_fit, rmean_fit, smean_fit, imean_fit, s1_fit, s2_fit, s3_fit = FPparams.x

            # Save the parameters to the list
            params_list.append(FPparams.x)

            # Set the new count of rejected galaxies
            badcount = badcountnew
            i += 1

            # Break from the loop if reject_outliers is set to false
            if reject_outliers == False:
                break

        # Save the cleaned sample
        print(f'Saving outlier-rejected sample...')
        df = data_fit
        # df.to_csv(outlier_output_filepath, index=False)

    # print the parameters
    print(params_list)