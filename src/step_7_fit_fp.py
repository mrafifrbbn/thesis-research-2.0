import os
import sys
import trace
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
from matplotlib import cm, ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Tuple

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

pvhub_dir = '/Users/mrafifrbbn/Documents/thesis/pvhub/'
if not pvhub_dir in sys.path: sys.path.append(pvhub_dir)
from pvhub import * # type: ignore

import emcee
import getdist
from getdist import plots, MCSamples

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
COMPLETENESS_SETTING = int(os.environ.get('COMPLETENESS_SETTING'))
# Add new data combinations here
NEW_SURVEY_LIST = (SURVEY_LIST + ['SDSS_LAMOST', 'ALL_COMBINED']) if SMIN_SETTING == 1 else SURVEY_LIST

# Create logging instance
logger = get_logger('fit_fp')

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss_lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/all_combined.csv')
}

OUTLIER_REJECT_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/sdss_lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/all_combined.csv')
}
create_parent_folder(OUTLIER_REJECT_OUTPUT_FILEPATH)

FP_FIT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fits.csv')
create_parent_folder(FP_FIT_FILEPATH)

MCMC_CHAIN_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/6dfgs_chain.npy'),
    'SDSS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/sdss_chain.npy'),
    'LAMOST': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/lamost_chain.npy'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/sdss_lamost_chain.npy'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/all_combined_chain.npy')
}
create_parent_folder(MCMC_CHAIN_OUTPUT_FILEPATH)

LIKELIHOOD_CORNERPLOT_IMG_FILEPATH = os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/three_surveys_likelihood.png')
create_parent_folder(LIKELIHOOD_CORNERPLOT_IMG_FILEPATH)

LIKELIHOOD_DIST_IMG_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/6dfgs.png'),
    'SDSS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/sdss.png'),
    'LAMOST': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/lamost.png'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/sdss_lamost.png'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/all_combined.png')
}

FP_SCATTER_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_scatter.csv')
create_parent_folder(FP_SCATTER_FILEPATH)

# User-defined variables
PVALS_CUT = 0.01
REJECT_OUTLIERS = True
PARAM_BOUNDARIES = [(1.2, 1.6), (-0.9, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.2, 3.5), (0.0, 0.06), (0.25, 0.45), (0.14, 0.25)]

# Set global random seed
np.random.seed(42)

def fit_FP(
    survey: str,
    df: pd.DataFrame,
    smin: float,
    param_boundaries: List[Tuple] = PARAM_BOUNDARIES,
    pvals_cut: float = PVALS_CUT,
    zmin: float = ZMIN,
    zmax: float = ZMAX,
    mag_high: float = MAG_HIGH,
    reject_outliers: bool = REJECT_OUTLIERS
          ) -> np.ndarray:
    
    logger.info(f"{'=' * 10} Fitting {survey} Fundamental Plane | Ngals = {len(df)} {'=' * 10}")
    
    # Re-apply the magnitude limit
    df = df[(df['j_m_ext'] - df['extinction_j']) <= mag_high]#.loc[:10, :]

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
    df['lmin'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_HIGH) / 5.0
    df['lmax'] = (SOLAR_MAGNITUDE['j'] + 5.0 * np.log10(1.0 + df["zhelio"].to_numpy()) + df["kcor_j"].to_numpy() + df["extinction_j"].to_numpy() + 10.0 - 2.5 * np.log10(2.0 * math.pi) + 5.0 * np.log10(d_H) - MAG_LOW) / 5.0

    # Calculate predicted logdistance-ratios
    d_z = sp.interpolate.splev(df['z_dist_est'].to_numpy(), dist_spline, der=0)
    df['logdist_pred'] = np.log10(d_z / d_H)
    df['r_true'] = df['r'] - df['logdist_pred']

    # Use Sn (default is not)
    if False: # Replace this with a variable in the future
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
    else:
        Sn = 1.0
        df['Sn'] = Sn

    # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
    data_fit = df
    badcount = len(df)
    # logger.info(len(data_fit), badcount)
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
            args=(0.0, data_fit["z_cmb"].to_numpy(), data_fit["r_true"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, smin, data_fit["lmin"].to_numpy(), data_fit["lmax"].to_numpy(), data_fit["C_m"].to_numpy()), maxiter=10000, tol=1.0e-6, workers=-1)
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

        # logger.info verbose
        logger.info(f"{'-' * 10} Iteration {i} {'-' * 10} | FP parameters: {FPparams.x.tolist()}")
        
        # Set the new count of rejected galaxies
        badcount = badcountnew
        i += 1
        
        # Break from the loop if reject_outliers is set to false
        if reject_outliers == False:
            break

    df = data_fit
    logger.info(f'Number of galaxies remaining: {len(df)}')
    
    return FPparams.x, df

def sample_likelihood(df: pd.DataFrame,
                      FP_params: np.ndarray, 
                      smin: float,
                      chain_output_filepath: str = None
                      ) -> np.ndarray:
    # The log-prior function for the FP parameters
    def log_prior(theta):
        a, b, rmean, smean, imean, sig1, sig2, sig3 = theta
        a_bound, b_bound, rmean_bound, smean_bound, imean_bound, s1_bound, s2_bound, s3_bound = PARAM_BOUNDARIES
        if a_bound[0] < a < a_bound[1] and b_bound[0] < b < b_bound[1] and rmean_bound[0] < rmean < rmean_bound[1] and smean_bound[0] < smean < smean_bound[1] and imean_bound[0] < imean < imean_bound[1] and s1_bound[0] < sig1 < s1_bound[1] and s2_bound[0] < sig2 < s2_bound[1] and s3_bound[0] < sig3 < s3_bound[1]:
            return 0.0
        else:
            return -np.inf

    # Calculate log-posterior distribution
    def log_probability(theta, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals=True, chi_squared_only=False):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp - FP_func(theta, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, lmin, lmax, C_m, sumgals, chi_squared_only)
    
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
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(0.0, z, r, s, i, dr, ds, di, Sn, smin, lmin, lmax, C_m, True, False)
    )
    sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True)

    # Flatten the chain and save as numpy array
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    if chain_output_filepath is not None:
        np.save(chain_output_filepath, flat_samples)

    # Get the mean values of the marginalized distribution
    x_ = flat_samples.T
    FP_params_mean = np.mean(x_, axis=1)

    return FP_params_mean
        
def generate_corner_plot() -> None:
    # Set border thickness
    mpl.rcParams['axes.linewidth'] = 2.0

    # 6dFGS data (mocks and previous values)
    samples_6df = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['6dFGS'])
    prev_vals_6df = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['6dFGS'].to_numpy()
    logger.info(f"Best fit values for 6dFGS: {prev_vals_6df}")

    # SDSS data (mocks and previous values)
    samples_sdss = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['SDSS'])
    prev_vals_sdss = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['SDSS'].to_numpy()
    logger.info(f"Best fit values for SDSS: {prev_vals_sdss}")

    # LAMOST data (mocks and previous values)
    samples_lamost = np.load(MCMC_CHAIN_OUTPUT_FILEPATH['LAMOST'])
    prev_vals_lamost = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc['LAMOST'].to_numpy()
    logger.info(f"Best fit values for LAMOST: {prev_vals_lamost}")

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

# def fit_likelihood() -> None:
#     def fit_and_plot(posteriors, fp_paramname, fp_labelname, axis):
#         def gaus(x, mu, sig):
#             return (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * ((x - mu) / sig)**2)
        
#         xdata = posteriors[fp_paramname]
#         y, x_edges = np.histogram(xdata, bins=N, density=True)
#         x = (x_edges[1:] + x_edges[:-1])/2
#         popt, pcov = curve_fit(gaus, x, y, p0=[np.mean(xdata), np.std(xdata)])
#         popt[1] = np.absolute(popt[1])
        
#         axis.hist(xdata, bins=N, density=True, alpha=0.5)
#         axis.plot(x, norm.pdf(x, loc=popt[0], scale=popt[1]), color='black')
#         axis.set_xlabel(fp_labelname, fontsize=15)
#         axis.set_title(r'%.4f' % popt[0] + ' $\pm$ %.4f' % popt[1])
#         logger.info(f"{'=' * 20} {fp_paramname} {'=' * 20}")
#         logger.info(f'Mean of {fp_paramname} = {round(popt[0], 4)}')
#         logger.info(f'Std of {fp_paramname} = {round(popt[1], 4)}')
#         fig.tight_layout(pad=1.0)
    
#     N = 50
#     for survey in NEW_SURVEY_LIST:
#         logger.info(f"Fitting the FP likelihood of {survey}")
#         post_dist = np.load(MCMC_CHAIN_OUTPUT_FILEPATH[survey]).T
#         posteriors = {
#             'a': post_dist[0],
#             'b': post_dist[1],
#             'rmean': post_dist[2],
#             'smean': post_dist[3],
#             'imean': post_dist[4],
#             'sigma1': post_dist[5],
#             'sigma2': post_dist[6],
#             'sigma3': post_dist[7]
#         }
        
#         fp_paramname_list = ['a', 'b', 'rmean', 'smean', 'imean', 'sigma1', 'sigma2', 'sigma3']
#         fp_labelname_list = [r'$a$', r'$b$', r'$\bar{r}$', r'$\bar{s}$', r'$\bar{\imath}$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
        
#         fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(GOLDEN_RATIO * 8, 8))
#         fig.delaxes(fig.axes[2])
        
#         for idx, ax in enumerate(fig.axes):
#             fit_and_plot(posteriors, fp_paramname_list[idx], fp_labelname_list[idx], ax)
#         fig.savefig(LIKELIHOOD_DIST_IMG_FILEPATH[survey])
#         logger.info('\n')

def calculate_fp_scatter() -> None:
    results = []
    for survey in NEW_SURVEY_LIST:
        df = pd.read_csv(OUTLIER_REJECT_OUTPUT_FILEPATH[survey])
        params = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc[survey].to_dict()
        a = params.get("a")
        b = params.get("b")
        sigma_1 = params.get("s1")
        
        # Calculate typical scatter in s
        err_spectro = df['es'].median()
        
        # Calculate the combined photometric error
        e_XFP = np.sqrt(df['er']**2 + (b * df['ei'])**2 + 2 * (-1) * np.absolute(b * df['er'] * df['ei']))
        err_photo = np.median(e_XFP)
        
        # Calculate the total intrinsic error in r
        sigma_r_int = sigma_1 * np.sqrt(1 + a**2 + b**2)
        
        # Calculate the total typical scatter in r
        r_scatter = np.sqrt((a * err_spectro)**2 + err_photo**2 + sigma_r_int**2)
        
        # Save everything in a dictionary
        scatter_dict = {
            "eps_s": err_spectro,
            "eps_photo": err_photo,
            "sigma_r_int": sigma_r_int,
            "r_scatter": r_scatter
        }
        results.append(scatter_dict)
        
    df = pd.DataFrame(results, index=NEW_SURVEY_LIST).round(decimals=4)
    df.to_csv(FP_SCATTER_FILEPATH, index=True)

def main() -> None:
    # try:
    logger.info(f'{"=" * 50}')
    logger.info(f'Fitting the Fundamental Plane using SMIN_SETTING = {SMIN_SETTING} | COMPLETENESS_SETTING = {COMPLETENESS_SETTING}...')
    logger.info(f'Sample selection constants:')
    logger.info(f'OMEGA_M = {OMEGA_M}')
    logger.info(f'smin = {SURVEY_VELDISP_LIMIT}')
    logger.info(f'MAG_LOW = {MAG_LOW}')
    logger.info(f'MAG_HIGH = {MAG_HIGH}')
    logger.info(f'ZMIN = {ZMIN}')
    logger.info(f'ZMAX = {ZMAX}')

    # Fit the FP for each survey and sample the likelihood
    FP_params = []
    for survey in NEW_SURVEY_LIST:
        # Get input filepath
        input_filepath = INPUT_FILEPATH[survey]
        df = pd.read_csv(input_filepath)

        # Velocity dispersion lower limit
        if SMIN_SETTING == 1:
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING]['6dFGS']
        else:
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]

        # FP parameter boundaries to search the maximum over
        param_boundaries = PARAM_BOUNDARIES

        params, df_fitted = fit_FP(
            survey=survey,
            df=df,
            smin=smin,
            param_boundaries=param_boundaries,
            reject_outliers=True
        )
        FP_params.append(params)

        # Save the cleaned sample
        output_filepath = OUTLIER_REJECT_OUTPUT_FILEPATH[survey]
        df_fitted.to_csv(output_filepath, index=False)

        # Sample the likelihood
        chain_output_filepath = MCMC_CHAIN_OUTPUT_FILEPATH[survey]
        params_mean = sample_likelihood(
            df=df_fitted,
            FP_params=params,
            smin=smin,
            chain_output_filepath=chain_output_filepath
            )
        # Calculate difference between likelihood sampling and diff. evo. algorithm
        params_diff = params - params_mean
        logger.info(f"Difference between evo algorithm and MCMC likelihood sampling = {params_diff}")
    
    # Convert the FP parameters to dataframe and save to artifacts folder
    logger.info("Saving the derived FP fits to artifacts folder...")
    FP_params = np.array(FP_params)
    FP_columns = ['a', 'b', 'rmean', 'smean', 'imean', 's1', 's2', 's3']
    pd.DataFrame(FP_params, columns=FP_columns, index=NEW_SURVEY_LIST).to_csv(FP_FIT_FILEPATH)
    
    # logger.info("Generating corner plot...")
    # generate_corner_plot()

    # logger.info("Fitting the marginalized distributions with Gaussian...")
    # fit_likelihood()

    # logger.info("Calculating the FP scatter...")
    # calculate_fp_scatter()

    # logger.info(f'Fitting the Fundamental Plane successful!')
    # except Exception as e:
    #     logger.error(f'Fitting the FP failed. Reason: {e}.')

if __name__ == '__main__':
    main()