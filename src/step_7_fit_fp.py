import os
import trace
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict

from utils.constants import *
from utils.CosmoFunc import *
from utils.logging_config import get_logger

import emcee
import getdist
from getdist import plots, MCSamples

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
# Add new data combinations here
NEW_SURVEY_LIST = (SURVEY_LIST + ['ALL_COMBINED']) if SMIN_SETTING == 1 else SURVEY_LIST

# Create logging instance
logger = get_logger('fit_fp')

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/all_combined.csv')
}

OUTLIER_REJECT_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/outlier_reject/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/outlier_reject/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/outlier_reject/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/outlier_reject/smin_setting_{SMIN_SETTING}/all_combined.csv')
}
create_parent_folder(OUTLIER_REJECT_OUTPUT_FILEPATH)

FP_FIT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fits.csv')
create_parent_folder(FP_FIT_FILEPATH)

MCMC_CHAIN_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/6dfgs_chain.npy'),
    'SDSS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/sdss_chain.npy'),
    'LAMOST': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/lamost_chain.npy'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/all_combined_chain.npy')
}
create_parent_folder(MCMC_CHAIN_OUTPUT_FILEPATH)

LIKELIHOOD_CORNERPLOT_IMG_FILEPATH = os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/three_surveys_likelihood.png')
create_parent_folder(LIKELIHOOD_CORNERPLOT_IMG_FILEPATH)

LIKELIHOOD_DIST_IMG_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/6dfgs.png'),
    'SDSS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/sdss.png'),
    'LAMOST': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/lamost.png'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/all_combined.png')
}

FP_SCATTER_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_scatter.csv')
create_parent_folder(FP_SCATTER_FILEPATH)

# User-defined variables
PVALS_CUT = 0.01
REJECT_OUTLIERS = True

def fit_FP(
    survey: str,
    input_filepath: str,
    outlier_output_filepath: str,
    smin: float,
    pvals_cut: float = PVALS_CUT,
    zmin: float = ZMIN,
    zmax: float = ZMAX,
    mag_high: float = MAG_HIGH,
    reject_outliers: bool = REJECT_OUTLIERS
          ) -> np.ndarray:
    
    # Set global random seed
    np.random.seed(42)
    
    logger.info(f"{'=' * 10} Fitting {survey} Fundamental Plane {'=' * 10}")
    df = pd.read_csv(input_filepath)
    
    # Re-apply the magnitude limit
    df = df[(df['j_m_ext'] - df['extinction_j']) <= mag_high]

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
    
    # Fitting the FP iteratively by rejecting galaxies with high chi-square (low p-values) in each iteration
    data_fit = df
    badcount = len(df)
    # logger.info(len(data_fit), badcount)
    is_converged = False
    i = 1
    
    while not is_converged:
        dz_cluster_fit = sp.interpolate.splev(data_fit["z_dist_est"].to_numpy(), dist_spline)
        Dlim = 10.0**((mag_high - (data_fit["j_m_ext"]-data_fit['extinction_j']).to_numpy() + 5.0 * np.log10(dz_cluster_fit) + 5.0*np.log10(1.0 + data_fit["zhelio"])) / 5.0)
        zlim = sp.interpolate.splev(Dlim, lumred_spline)

        Snfit = np.where(zlim >= zmax, 1.0, np.where(zlim <= zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

        # The range of the FP parameters' values
        avals, bvals = (1.0, 1.8), (-1.0, -0.5)
        rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.5), (3.0, 3.5)
        s1vals, s2vals, s3vals = (0., 0.3), (0.1, 0.5), (0.1, 0.3)

        # Fit the FP parameters
        FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
            args=(0.0, data_fit["z_cmb"].to_numpy(), data_fit["r"].to_numpy(), data_fit["s"].to_numpy(), data_fit["i"].to_numpy(), data_fit["er"].to_numpy(), data_fit["es"].to_numpy(), data_fit["ei"].to_numpy(), Snfit, smin), maxiter=10000, tol=1.0e-6)
        # Calculate the chi-squared 
        chi_squared = Sn*FP_func(FPparams.x, 0.0, df["z_cmb"].to_numpy(), df["r"].to_numpy(), df["s"].to_numpy(), df["i"].to_numpy(), df["er"].to_numpy(), df["es"].to_numpy(), df["ei"].to_numpy(), Sn, smin, sumgals=False, chi_squared_only=True)[0]

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
        
        # Break from the loop if reject_outliers is set to false
        if reject_outliers == False:
            break

    # Save the cleaned sample
    logger.info(f'Saving outlier-rejected sample...')
    df = data_fit
    df.to_csv(outlier_output_filepath, index=False)
    
    return FPparams.x

def sample_likelihood() -> None:
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
            return lp - FP_func(theta, 0., z, r, s, i, dr, ds, di, Sn, smin, sumgals=True, chi_squared_only=False)
    
    for survey in NEW_SURVEY_LIST:
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

        # Velocity dispersion lower limit
        if survey == 'ALL_COMBINED':
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING]['6dFGS']
        else:
            smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]

        # Get some redshift-distance lookup tables
        red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
        # The comoving distance to each galaxy using group redshift as distance indicator
        dz = sp.interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline, der=0) 

        # (1+z) factor because we use luminosity distance
        Vmin = (1.0 + ZMIN)**3 * sp.interpolate.splev(ZMIN, dist_spline)**3
        Vmax = (1.0 + ZMAX)**3 * sp.interpolate.splev(ZMAX, dist_spline)**3
        # Maximum (luminosity) distance the galaxy can be observed given MAG_HIGH (survey limiting magnitude)
        Dlim = 10.0**((MAG_HIGH - (df["j_m_ext"] - df['extinction_j']) + 5.0 * np.log10(dz) + 5.0 * np.log10(1.0 + df["zhelio"])) / 5.0)    
        # Find the corresponding maximum redshift
        zlim = sp.interpolate.splev(Dlim, lumred_spline)
        Sn = np.where(zlim >= ZMAX, 1.0, np.where(zlim <= ZMIN, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin)))

        # Load the best-fit parameters
        FP_params = pd.read_csv(FP_FIT_FILEPATH, index_col=0).loc[survey].to_numpy()

        # Specify the initial guess, the number of walkers, and dimensions
        pos = FP_params + 1e-2 * np.random.randn(16, 8)
        nwalkers, ndim = pos.shape

        # Flat prior boundaries (same order as FP_params)
        param_boundaries = [(1.0, 1.8), (-1.0, -0.5), (-0.5, 0.5), (1.8, 2.5), (2.8, 3.5), (0.0, 0.3), (0.1, 0.5), (0.1, 0.3)]

        # Run the MCMC
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(param_boundaries, 0., z, r, s, i, dr, ds, di, Sn, dz, 0., 0., smin, True, False)
        )
        sampler.run_mcmc(pos, 10000, progress=True, skip_initial_state_check=True)

        # Flatten the chain and save as numpy array
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        np.save(MCMC_CHAIN_OUTPUT_FILEPATH[survey], flat_samples)
        
def generate_corner_plot() -> None:
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

def fit_likelihood() -> None:
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
    for survey in NEW_SURVEY_LIST:
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
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(GOLDEN_RATIO * 8, 8))
        fig.delaxes(fig.axes[2])
        
        for idx, ax in enumerate(fig.axes):
            fit_and_plot(posteriors, fp_paramname_list[idx], fp_labelname_list[idx], ax)
        fig.savefig(LIKELIHOOD_DIST_IMG_FILEPATH[survey])
        logger.info('\n')

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
    try:
        logger.info(f'{"=" * 50}')
        logger.info(f'Fitting the Fundamental Plane using SMIN_SETTING = {SMIN_SETTING}...')
        logger.info(f'Sample selection constants:')
        logger.info(f'OMEGA_M = {OMEGA_M}')
        logger.info(f'smin = {SURVEY_VELDISP_LIMIT}')
        logger.info(f'MAG_LOW = {MAG_LOW}')
        logger.info(f'MAG_HIGH = {MAG_HIGH}')
        logger.info(f'ZMIN = {ZMIN}')
        logger.info(f'ZMAX = {ZMAX}')

        # Fit the FP for each survey
        FP_params = []
        for survey in NEW_SURVEY_LIST:
            # Get input filepath
            input_filepath = INPUT_FILEPATH[survey]

            # Get output filepath
            output_filepath = OUTLIER_REJECT_OUTPUT_FILEPATH[survey]

            # Velocity dispersion lower limit
            if survey == 'ALL_COMBINED':
                smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING]['6dFGS']
            else:
                smin = SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]

            params = fit_FP(
                survey=survey,
                input_filepath=input_filepath,
                outlier_output_filepath=output_filepath,
                smin=smin
            )
            FP_params.append(params)
        
        # Convert the FP parameters to dataframe and save to artifacts folder
        logger.info("Saving the derived FP fits to artifacts folder...")
        FP_params = np.array(FP_params)
        FP_columns = ['a', 'b', 'rmean', 'smean', 'imean', 's1', 's2', 's3']
        pd.DataFrame(FP_params, columns=FP_columns, index=NEW_SURVEY_LIST).to_csv(FP_FIT_FILEPATH)
        
        logger.info("Sampling the likelihood with MCMC...")
        sample_likelihood()
        
        logger.info("Generating corner plot...")
        generate_corner_plot()

        logger.info("Fitting the marginalized distributions with Gaussian...")
        fit_likelihood()

        logger.info("Calculating the FP scatter...")
        calculate_fp_scatter()

        logger.info(f'Fitting the Fundamental Plane successful!')
    except Exception as e:
        logger.error(f'Fitting the FP failed. Reason: {e}.')

if __name__ == '__main__':
    main()