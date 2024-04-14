import os
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

import emcee
import getdist
from getdist import plots, MCSamples

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
USE_6dFGS_OFFSET = True if os.environ.get('USE_6dFGS_OFFSET').lower() == 'true' else False

# Create logging instance
logger = get_logger('fit_fp')

if not USE_6dFGS_OFFSET:
    INPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/6dfgs.csv'),
        'SDSS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/sdss.csv'),
        'LAMOST': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/lamost.csv')
    }

    OUTLIER_REJECT_OUTPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/6dfgs.csv'),
        'SDSS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/sdss.csv'),
        'LAMOST': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/lamost.csv')
    }

    FP_FIT_FILEPATH = os.path.join(ROOT_PATH, 'artifacts/fp_fit/fp_fits.csv')

    MCMC_CHAIN_OUTPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'artifacts/fp_fit/6dfgs_chain.npy'),
        'SDSS': os.path.join(ROOT_PATH, 'artifacts/fp_fit/sdss_chain.npy'),
        'LAMOST': os.path.join(ROOT_PATH, 'artifacts/fp_fit/lamost_chain.npy')
    }

    LIKELIHOOD_CORNERPLOT_IMG_FILEPATH = os.path.join(ROOT_PATH, 'img/fp_fit/three_surveys_likelihood.png')

    LIKELIHOOD_DIST_IMG_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'img/fp_fit/6dfgs.png'),
        'SDSS': os.path.join(ROOT_PATH, 'img/fp_fit/sdss.png'),
        'LAMOST': os.path.join(ROOT_PATH, 'img/fp_fit/lamost.png')
    }
else:
    INPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/use_offset/6dfgs.csv'),
        'SDSS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/use_offset/sdss.csv'),
        'LAMOST': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/use_offset/lamost.csv')
    }

    OUTLIER_REJECT_OUTPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/use_offset/6dfgs.csv'),
        'SDSS': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/use_offset/sdss.csv'),
        'LAMOST': os.path.join(ROOT_PATH, 'data/foundation/fp_sample/outlier_reject/use_offset/lamost.csv')
    }

    FP_FIT_FILEPATH = os.path.join(ROOT_PATH, 'artifacts/fp_fit/use_offset/fp_fits.csv')

    MCMC_CHAIN_OUTPUT_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'artifacts/fp_fit/use_offset/6dfgs_chain.npy'),
        'SDSS': os.path.join(ROOT_PATH, 'artifacts/fp_fit/use_offset/sdss_chain.npy'),
        'LAMOST': os.path.join(ROOT_PATH, 'artifacts/fp_fit/use_offset/lamost_chain.npy')
    }

    LIKELIHOOD_CORNERPLOT_IMG_FILEPATH = os.path.join(ROOT_PATH, 'img/fp_fit/use_offset/three_surveys_likelihood.png')

    LIKELIHOOD_DIST_IMG_FILEPATH = {
        '6dFGS': os.path.join(ROOT_PATH, 'img/fp_fit/use_offset/6dfgs.png'),
        'SDSS': os.path.join(ROOT_PATH, 'img/fp_fit/use_offset/sdss.png'),
        'LAMOST': os.path.join(ROOT_PATH, 'img/fp_fit/use_offset/lamost.png')
    }

# Grab 6dFGS offset
totoff = pd.read_csv(os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/totoffs.csv'))
if not USE_6dFGS_OFFSET:
    off_6df = 0.0
else:
    off_6df = totoff.loc[0, ['off_6df']].values[0]

# Sample selection constants
# The magnitude limit and velocity dispersion limits (very important!), and Omega_m (less important)
omega_m = 0.3121
smin = np.log10(112) - off_6df
mag_low = 8.0
mag_high = 13.65
zmin = 3000.0 / LIGHTSPEED
zmax = 16120. / LIGHTSPEED

# The likelihood function for the Fundamental Plane
def FP_func(params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, dz, kr, Ar, smin, sumgals=True, chi_squared_only=False):

    # Trial FP parameters
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params 
    # The additional parameter assumed to be zero
    k = 0.0

    # These are the components of the eigenvectors and their magnitudes
    fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
    norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
    # The components of the scatter matrix
    dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
    sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
    sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
    sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
    sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
    sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
    sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2
    sigma_cov = np.array([[sigmar2, sigmars, sigmari], [sigmars, sigmas2, sigmasi], [sigmari, sigmasi, sigmai2]])

    # Compute the chi-squared and determinant (quickly!)
    # The component of the covariance matrix
    cov_r = err_r**2 + np.log10(1.0 + 300.0/(LIGHTSPEED*z_obs))**2 + sigmar2
    cov_s = err_s**2 + sigmas2
    cov_i = err_i**2 + sigmai2
    cov_ri = -1.0*err_r*err_i + sigmari

    # Minors of the covariance matrix
    A = cov_s*cov_i - sigmasi**2
    B = sigmasi*cov_ri - sigmars*cov_i
    C = sigmars*sigmasi - cov_s*cov_ri
    E = cov_r*cov_i - cov_ri**2
    F = sigmars*cov_ri - cov_r*sigmasi
    I = cov_r*cov_s - sigmars**2

    # Using the mean values rbar, sbar, ibar as the center
    sdiff, idiff = s - smean, i - imean
    rnew = r - np.tile(logdists, (len(r), 1)).T
    rdiff  = rnew - rmean

    # Determinant of the covariance matrix
    det = cov_r*A + sigmars*B + cov_ri*C
    log_det = np.log(det)/Sn

    # The chi-squared term of the likelihood function
    chi_squared = (A*rdiff**2 + E*sdiff**2 + I*idiff**2 + 2.0*rdiff*(B*sdiff + C*idiff) + 2.0*F*sdiff*idiff)/(det*Sn)

    # Compute the FN term for the Scut only
    delta = (A*F**2 + I*B**2 - 2.0*B*C*F)/det
    FN = np.log(0.5 * special.erfc(np.sqrt(E/(2.0*(det+delta)))*(smin-smean)))/Sn

    if chi_squared_only:
        return chi_squared
    elif sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0*FN)
    else:
        return 0.5 * (chi_squared + log_det)

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

def fit_FP():
    # Set global random seed
    np.random.seed(42)
    
    # List to store FP parameters
    FP_params = []
    
    for survey in SURVEY_LIST:
        logger.info(f"{'=' * 10} Fitting {survey} Fundamental Plane {'=' * 10}")
        df = pd.read_csv(INPUT_FILEPATH[survey])

        # p-value upper limit, reject galaxies with p-value lower than this limit
        pvals_cut = 0.01

        # Get some redshift-distance lookup tables
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        # The comoving distance to each galaxy using group redshift as distance indicator
        dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0) 

        # (1+z) factor because we use luminosity distance
        Vmin = (1.0 + zmin)**3 * sp.interpolate.splev(zmin, dist_spline)**3
        Vmax = (1.0 + zmax)**3 * sp.interpolate.splev(zmax, dist_spline)**3
        # Maximum (luminosity) distance the galaxy can be observed given mag_high (survey limiting magnitude)
        Dlim = 10.0**((mag_high - (df["j_m_ext"] - df['extinction_j']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)    
        # Find the corresponding maximum redshift
        zlim = sp.interpolate.splev(Dlim, lumred_spline)
        Sn = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

        # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
        data_fit = df
        badcount = len(df)
        # logger.info(len(data_fit), badcount)
        is_converged = False
        i = 1
        
        while not is_converged:
            dz_cluster_fit = sp.interpolate.splev(data_fit["z_dist_est"].to_numpy(), dist_spline)
            Dlim = 10.0**((mag_high - (data_fit["j_m_ext"]-data_fit['extinction_j']).to_numpy() + 5.0 * np.log10(dz_cluster_fit) + 5.0*np.log10(1.0 + data_fit["zhelio"]))/5.0)
            zlim = sp.interpolate.splev(Dlim, lumred_spline)

            Snfit = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

            # The range of the FP parameters' values
            avals, bvals = (1.3, 1.8), (-1.0, -0.5)
            rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.5), (3.0, 3.5)
            s1vals, s2vals, s3vals = (0., 0.3), (0.1, 0.5), (0.1, 0.3)

            # Fit the FP parameters
            FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                args=(0.0, data_fit["z_cmb"].to_numpy(), data_fit["r"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, dz, data_fit["kcor_j"].to_numpy(), data_fit["extinction_j"].to_numpy(), smin), seed=42, maxiter=10000, tol=1.0e-6)
            # Calculate the chi-squared 
            chi_squared = Sn*FP_func(FPparams.x, 0.0, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, dz, df["kcor_j"].to_numpy(), df["extinction_j"].to_numpy(), smin, sumgals=False, chi_squared_only=True)[0]

            # Calculate the p-value (x,dof)
            pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(df) - 8.0))
            # Reject galaxies with p-values < pvals_cut
            data_fit = df.drop(df[pvals < pvals_cut].index).reset_index(drop=True)
            # Count the number of rejected galaxies
            badcountnew = len(np.where(pvals < pvals_cut)[0])
            # Converged if the number of rejected galaxies in this iteration is the same as previous iteration
            is_converged = True if badcount == badcountnew else False

            # logger.info verbose
            logger.info(f"{'-' * 10} Iteration {i} {'-' * 10}")
            logger.info(FPparams.x.tolist())
            a_fit, b_fit, rmean_fit, smean_fit, imean_fit, s1_fit, s2_fit, s3_fit = FPparams.x
            logger.info(f"a = {round(a_fit, 5)}")
            logger.info(f"b = {round(b_fit, 5)}")
            logger.info(f"rmean = {round(rmean_fit, 5)}")
            logger.info(f"smean = {round(smean_fit, 5)}")
            logger.info(f"imean = {round(imean_fit, 5)}")
            logger.info(f"s1 = {round(s1_fit, 5)}")
            logger.info(f"s2 = {round(s2_fit, 5)}")
            logger.info(f"s3 = {round(s3_fit, 5)}")
            logger.info(f"Data count = {len(data_fit)}")
            logger.info(f"Chi-squared = {sp.stats.chi2.isf(0.01, np.sum(chi_squared)/(len(df) - 8.0))}")
            logger.info(f"Outlier count = {badcount}")
            logger.info(f"New outlier count = {badcountnew}")
            logger.info(f"Converged = {is_converged}")
            
            # Set the new count of rejected galaxies
            badcount = badcountnew
            i += 1

        # Store the FP parameters
        FP_params.append(FPparams.x)
        
        # Save the cleaned sample
        df = data_fit
        df.to_csv(OUTLIER_REJECT_OUTPUT_FILEPATH[survey], index=False)
        logger.info('\n')
        
    # Convert the FP parameters to dataframe and save to artifacts folder
    FP_params = np.array(FP_params)
    FP_columns = ['a', 'b', 'rmean', 'smean', 'imean', 's1', 's2', 's3']
    pd.DataFrame(FP_params, columns=FP_columns, index=SURVEY_LIST).to_csv(FP_FIT_FILEPATH)

def sample_likelihood():
    # The log-prior function for the FP parameters
    def log_prior(theta, param_boundaries):
        a, b, rmean, smean, imean, sig1, sig2, sig3 = theta
        a_bound, b_bound, rmean_bound, smean_bound, imean_bound, s1_bound, s2_bound, s3_bound = param_boundaries
        if a_bound[0] < a < a_bound[1] and b_bound[0] < b < b_bound[1] and rmean_bound[0] < rmean < rmean_bound[1] and smean_bound[0] < smean < smean_bound[1] and imean_bound[0] < imean < imean_bound[1] and s1_bound[0] < sig1 < s1_bound[1] and s2_bound[0] < sig2 < s2_bound[1] and s3_bound[0] < sig3 < s3_bound[1]:
            return 0.0
        else:
            return -np.inf

    # Calculate log-posterior distribution
    def log_probability(theta, param_boundaries, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, dz, kr, Ar, smin, sumgals=True, chi_squared_only=False):
        lp = log_prior(theta, param_boundaries)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - FP_func(theta, 0., z, r, s, i, dr, ds, di, Sn, dz, 0., 0., smin, sumgals=True, chi_squared_only=False)
    
    for survey in SURVEY_LIST:
        # Load the outlier-rejected data
        df = pd.read_csv(OUTLIER_REJECT_OUTPUT_FILEPATH[survey])

        z = df['z_dist_est'].to_numpy()
        r = df['r'].to_numpy()
        s = df['s'].to_numpy()
        i = df['i'].to_numpy()
        dr = df['er'].to_numpy()
        ds = df['es'].to_numpy()
        di = df['ei'].to_numpy()
        mag_j = df['j_m_ext'].to_numpy()
        A_j = df['extinction_j'].to_numpy()

        # Get some redshift-distance lookup tables
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        # The comoving distance to each galaxy using group redshift as distance indicator
        dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0) 

        # (1+z) factor because we use luminosity distance
        Vmin = (1.0 + zmin)**3 * sp.interpolate.splev(zmin, dist_spline)**3
        Vmax = (1.0 + zmax)**3 * sp.interpolate.splev(zmax, dist_spline)**3
        # Maximum (luminosity) distance the galaxy can be observed given mag_high (survey limiting magnitude)
        Dlim = 10.0**((mag_high - (df["j_m_ext"] - df['extinction_j']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)    
        # Find the corresponding maximum redshift
        zlim = sp.interpolate.splev(Dlim, lumred_spline)
        Sn = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

        # Load the best-fit parameters
        FP_params = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc[survey].to_numpy()

        # Specify the initial guess, the number of walkers, and dimensions
        pos = FP_params + 1e-2 * np.random.randn(16, 8)
        nwalkers, ndim = pos.shape

        # Flat prior boundaries (same order as FP_params)
        param_boundaries = [(1.3, 1.8), (-1.0, -0.5), (-0.5, 0.5), (1.8, 2.5), (2.8, 3.5), (0.0, 0.3), (0.1, 0.5), (0.1, 0.3)]

        # Run the MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(param_boundaries, 0., z, r, s, i, dr, ds, di, Sn, dz, 0., 0., smin, True, False)
        )
        sampler.run_mcmc(pos, 10000, progress=True, skip_initial_state_check=True)

        # Flatten the chain and save as numpy array
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        np.save(MCMC_CHAIN_OUTPUT_FILEPATH[survey], flat_samples)
        
def generate_corner_plot():
    # Set border thickness
    mpl.rcParams['axes.linewidth'] = 2.0

    # 6dFGS data (mocks and previous values)
    samples_6df = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['6dFGS'])
    prev_vals_6df = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['6dFGS'].to_numpy()

    # SDSS data (mocks and previous values)
    samples_sdss = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['SDSS'])
    prev_vals_sdss = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['SDSS'].to_numpy()

    # LAMOST data (mocks and previous values)
    samples_lamost = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['LAMOST'])
    prev_vals_lamost = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['LAMOST'].to_numpy()

    # parameter names
    names = [r'$a$', r'$b$', r'$\bar{r}$', r'$\bar{s}$', r'$\bar{\imath}$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']

    samples1 = getdist.MCSamples(samples=samples_6df, names=names, label='6dFGS')
    samples2 = getdist.MCSamples(samples=samples_sdss, names=names, label='SDSS')
    samples3 = getdist.MCSamples(samples=samples_lamost, names=names, label='LAMOST')

    # Triangle plot
    g = getdist.plots.get_subplot_plotter()
    g.settings.legend_fontsize = 25
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 20
    g.triangle_plot([samples1, samples2, samples3], filled=True)

    ndim = 8
    for i in range(ndim):    
        for j in range(ndim):
            if j<=i:
                ax = g.subplots[i,j]
                ax.axvline(prev_vals_6df[j], color='grey', ls='--', alpha=0.5)
                ax.axvline(prev_vals_sdss[j], color='red', ls='--', alpha=0.5)
                ax.axvline(prev_vals_lamost[j], color='blue', ls='--', alpha=0.5)

                if i != j:
                    ax.axhline(prev_vals_6df[i], color='grey', ls='--', alpha=0.5)
                    ax.axhline(prev_vals_sdss[i], color='red', ls='--', alpha=0.5)
                    ax.axhline(prev_vals_lamost[i], color='blue', ls='--', alpha=0.5)

    g.export(LIKELIHOOD_CORNERPLOT_IMG_FILEPATH, dpi=300)

def fit_likelihood():
    def fit_and_plot(posteriors, fp_paramname, fp_labelname, axis):
        def gaus(x, mu, sig):
            return (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * ((x - mu) / sig)**2)
        
        xdata = posteriors[fp_paramname]
        y, x_edges = np.histogram(xdata, bins=N, density=True)
        x = (x_edges[1:] + x_edges[:-1])/2
        popt, pcov = curve_fit(gaus, x, y, p0=[np.mean(xdata), np.std(xdata)])
        popt[1] = np.absolute(popt[1])
        
        axis.hist(xdata, bins=N, density=True, alpha=0.5)
        axis.plot(x, norm.pdf(x, loc=popt[0], scale=popt[1]), color='black')
        axis.set_xlabel(fp_labelname, fontsize=15)
        axis.set_title(r'%.4f' % popt[0] + ' $\pm$ %.4f' % popt[1])
        logger.info(f"{'=' * 20} {fp_paramname} {'=' * 20}")
        logger.info(f'Mean of {fp_paramname} = {round(popt[0], 4)}')
        logger.info(f'Std of {fp_paramname} = {round(popt[1], 4)}')
        fig.tight_layout(pad=1.0)
    
    N = 50
    for survey in SURVEY_LIST:
        logger.info(f"Fitting the FP likelihood of {survey}")
        post_dist = np.load(MCMC_CHAIN_OUTPUT_FILEPATH[survey]).T
        posteriors = {
            'a': post_dist[0],
            'b': post_dist[1],
            'rmean': post_dist[2],
            'smean': post_dist[3],
            'imean': post_dist[4],
            'sigma1': post_dist[5],
            'sigma2': post_dist[6],
            'sigma3': post_dist[7]
        }
        
        fp_paramname_list = ['a', 'b', 'rmean', 'smean', 'imean', 'sigma1', 'sigma2', 'sigma3']
        fp_labelname_list = [r'$a$', r'$b$', r'$\bar{r}$', r'$\bar{s}$', r'$\bar{\imath}$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
        
        golden_ratio = 1.618
        height = 8
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(golden_ratio * height, height))
        fig.delaxes(fig.axes[2])
        
        for idx, ax in enumerate(fig.axes):
            fit_and_plot(posteriors, fp_paramname_list[idx], fp_labelname_list[idx], ax)
        fig.savefig(LIKELIHOOD_DIST_IMG_FILEPATH[survey])
        logger.info('\n')

def main():
    try:
        logger.info({f'{"=" * 20}'})
        logger.info(f'Fitting the Fundamental Plane...')
        logger.info(f'Sample selection constants:')
        logger.info(f'omega_m = {omega_m}')
        logger.info(f'smin = {smin}')
        logger.info(f'mag_low = {mag_low}')
        logger.info(f'mag_high = {mag_high}')
        logger.info(f'zmin = {zmin}')
        logger.info(f'zmax = {zmax}')
        # fit_FP()
        
        # sample_likelihood()
        
        # generate_corner_plot()

        fit_likelihood()

    except Exception as e:
        logger.error(f'Fitting the FP failed. Reason: {e}.')

if __name__ == '__main__':
    main()