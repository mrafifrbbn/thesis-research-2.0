import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

from utils.constants import *

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('veldisp_calibration')

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/lamost.csv')
}

OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/lamost.csv')
}

VELDISP_ORI_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/repeat_ori.csv')
VELDISP_SCALED_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/repeat_scaled.csv')
VELDISP_TOTOFF_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/totoffs.csv')

def get_common_galaxies():
    '''
    A function to find galaxies with repeat velocity dispersion measurements.
    '''
    try:
        # Grab all of the data and only use their 2MASS and survey ID's, also grab the velocity dispersion measurements
        logger.info('Opening r, s, i data...')
        data_6df = pd.read_csv(INPUT_FILEPATH['6dFGS'])[['tmass', '_6dFGS', 's', 'es']]
        data_6df_id = data_6df[['tmass','_6dFGS']]
        data_6df_veldisp = data_6df[['tmass', 's', 'es']].rename(columns={'s':'s_6df','es':'es_6df'})

        data_sdss = pd.read_csv(INPUT_FILEPATH['SDSS'])[['tmass', 'objid', 's', 'es']]
        data_sdss_id = data_sdss[['tmass','objid']]
        data_sdss_veldisp = data_sdss[['tmass', 's', 'es']].rename(columns={'s':'s_sdss','es':'es_sdss'})

        data_lamost = pd.read_csv(INPUT_FILEPATH['LAMOST'])[['tmass', 'obsid', 's', 'es']]
        data_lamost_id = data_lamost[['tmass','obsid']]
        data_lamost_veldisp = data_lamost[['tmass', 's', 'es']].rename(columns={'s':'s_lamost','es':'es_lamost'})

        # Find galaxies with repeat measurements for every survey combination
        logger.info('Finding common galaxies...')
        repeat_6df_sdss = pd.merge(data_6df_id, data_sdss_id, on='tmass', how='inner')
        repeat_6df_lamost = pd.merge(data_6df_id, data_lamost_id, on='tmass', how='inner')
        repeat_sdss_lamost = pd.merge(data_sdss_id, data_lamost_id, on='tmass', how='inner')

        # Find galaxies with 3 repeat measurements (observed in all 3 surveys)
        repeat_3 = pd.merge(repeat_6df_sdss, repeat_sdss_lamost, on='tmass', how='inner')\
                            .drop('objid_y',axis=1)\
                            .rename(columns={'objid_x':'objid'})
        logger.info(f'Number of common galaxies in 6dFGS-SDSS-LAMOST = {len(repeat_3)}')

        # Find galaxies with 2 repeat measurements by removing the repeat 3 galaxies from all of the repeat measurements above
        repeat_6df_sdss_only = repeat_6df_sdss[~repeat_6df_sdss['tmass'].isin(repeat_3['tmass'].tolist())]
        logger.info(f'Number of common galaxies in 6dFGS-SDSS only = {len(repeat_6df_sdss_only)}')
        repeat_6df_lamost_only = repeat_6df_lamost[~repeat_6df_lamost['tmass'].isin(repeat_3['tmass'].tolist())]
        logger.info(f'Number of common galaxies in 6dFGS-LAMOST only = {len(repeat_6df_lamost_only)}')
        repeat_sdss_lamost_only = repeat_sdss_lamost[~repeat_sdss_lamost['tmass'].isin(repeat_3['tmass'].tolist())]
        logger.info(f'Number of common galaxies in SDSS-LAMOST only = {len(repeat_sdss_lamost_only)}')

        # Create the third survey id name filled with NaN values (so we can concatenate the dataframes later)
        repeat_6df_sdss_only.loc[:, ['obsid']] = np.nan
        repeat_6df_lamost_only.loc[:, ['objid']] = np.nan
        repeat_sdss_lamost_only.loc[:, ['_6dFGS']] = np.nan

        # Concatenate the dataframes
        df = pd.concat([repeat_3, repeat_6df_sdss_only, repeat_6df_lamost_only, repeat_sdss_lamost_only])
        # Join with the velocity dispersion measurements
        df = df.merge(data_6df_veldisp, how='left', on='tmass')\
            .merge(data_sdss_veldisp, how='left', on='tmass')\
            .merge(data_lamost_veldisp, how='left', on='tmass')
        logger.info(f'Number of unique common galaxies in 6dFGS-SDSS-LAMOST = {len(df)}')

        # Save the dataframe
        df.to_csv(VELDISP_ORI_OUTPUT_FILEPATH, index=False)

        return df

    except Exception as e:
        logger.error(f'Finding common galaxies failed. Reason: {e}.')

def update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, survey, k_sdss=1.0, k_lamost=1.0, sigma_clip=3.0, convergence_tol=0.005):
    '''
    A function to calculate the error scaling for SDSS and LAMOST.
    '''
    try:
        logger.info(f'Updating error scaling for {survey}. Number of galaxies used = {len(s_sdss)}')
        # Scale the errors
        es_sdss_scaled = k_sdss * es_sdss
        es_lamost_scaled = k_lamost * es_lamost
        
        # Calculate the pairwise statistics
        epsilon = (s_sdss - s_lamost) / np.sqrt(es_sdss_scaled**2 + es_lamost_scaled**2)
        
        # Apply sigma clipping before calculating the new error scaling
        logger.info(f'Applying {sigma_clip} sigma clipping...')
        sigma_clip_filter = np.logical_and(~np.isnan(epsilon), np.absolute(epsilon) < sigma_clip)
        es_sdss_clipped = es_sdss_scaled[sigma_clip_filter]
        es_lamost_clipped = es_lamost_scaled[sigma_clip_filter]
        epsilon_clipped = epsilon[sigma_clip_filter]
        logger.info(f'Number of comparisons remaining = {len(epsilon_clipped)}.')
        
        # Update the error scaling and check convergence
        N = len(epsilon_clipped)
        rms_sdss = (1 / N) * np.sum(es_sdss_clipped**2)
        rms_lamost = (1 / N) * np.sum(es_lamost_clipped**2)
        f2 = np.std(epsilon_clipped)**2
        
        if survey == 'SDSS':
            k_new = np.sqrt(f2 + (f2 - 1) * (rms_lamost / rms_sdss))
            k_updated = k_sdss * k_new
            is_convergent = np.absolute((k_updated - k_sdss) / k_updated) * 100 < convergence_tol
        elif survey == 'LAMOST':
            k_new = np.sqrt(f2 + (f2 - 1)*(rms_sdss / rms_lamost))
            k_updated = k_lamost * k_new
            is_convergent = np.absolute((k_updated - k_lamost) / k_updated) * 100 < convergence_tol
        
        logger.info(f'New scaling for {survey} = {k_updated}.')
        return k_updated, is_convergent
    except Exception as e:
        logger.error(f'Updating scaling for {survey} failed. Reason: {e}.')

def get_error_scaling_old_method(df_repeat, sigma_clip=5.0, max_iter=10, sdss_first=True):
    '''
    A function to obtain the error scaling
    '''
    def update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, survey, k_sdss=1.0, k_lamost=1.0, sigma_clip=3.0, convergence_tol=0.005):
        '''
        A function to calculate the error scaling for SDSS and LAMOST.
        '''
        logger.info(f'Updating error scaling for {survey}. Number of galaxies used = {len(s_sdss)}')
        # Scale the errors
        es_sdss_scaled = k_sdss * es_sdss
        es_lamost_scaled = k_lamost * es_lamost

        # Calculate the pairwise statistics
        epsilon = (s_sdss - s_lamost) / np.sqrt(es_sdss_scaled**2 + es_lamost_scaled**2)

        # Apply sigma clipping before calculating the new error scaling
        logger.info(f'Applying {sigma_clip} sigma clipping...')
        sigma_clip_filter = np.logical_and(~np.isnan(epsilon), np.absolute(epsilon) < sigma_clip)
        es_sdss_clipped = es_sdss_scaled[sigma_clip_filter]
        es_lamost_clipped = es_lamost_scaled[sigma_clip_filter]
        epsilon_clipped = epsilon[sigma_clip_filter]
        logger.info(f'Number of comparisons remaining = {len(epsilon_clipped)}.')

        # Update the error scaling and check convergence
        N = len(epsilon_clipped)
        rms_sdss = (1 / N) * np.sum(es_sdss_clipped**2)
        rms_lamost = (1 / N) * np.sum(es_lamost_clipped**2)
        f2 = np.std(epsilon_clipped)**2

        if survey == 'SDSS':
            k_new = np.sqrt(f2 + (f2 - 1) * (rms_lamost / rms_sdss))
            k_updated = k_sdss * k_new
            is_convergent = np.absolute((k_updated - k_sdss) / k_updated) * 100 < convergence_tol
        elif survey == 'LAMOST':
            k_new = np.sqrt(f2 + (f2 - 1) * (rms_sdss / rms_lamost))
            k_updated = k_lamost * k_new
            is_convergent = np.absolute((k_updated - k_lamost) / k_updated) * 100 < convergence_tol

        logger.info(f'New scaling for {survey} = {k_updated}.')
        return k_updated, is_convergent

    # SDSS and LAMOST veldisp
    s_sdss = df_repeat['s_sdss'].to_numpy()
    es_sdss = df_repeat['es_sdss'].to_numpy()
    s_lamost = df_repeat['s_lamost'].to_numpy()
    es_lamost = df_repeat['es_lamost'].to_numpy()
    
    # Initial scalings for SDSS and LAMOST
    k_sdss = 1.0
    k_lamost = 1.0
    
    # Find the error scalings
    for i in range(max_iter):
        if sdss_first:
            # Update SDSS error
            k_sdss, is_sdss_convergent = update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, 'SDSS', k_sdss, k_lamost, sigma_clip)
            # Update LAMOST error
            k_lamost, is_lamost_convergent = update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, 'LAMOST', k_sdss, k_lamost, sigma_clip)
        else:
            # Update LAMOST error
            k_lamost, is_lamost_convergent = update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, 'LAMOST', k_sdss, k_lamost, sigma_clip)
            # Update SDSS error
            k_sdss, is_sdss_convergent = update_error_scaling(s_sdss, es_sdss, s_lamost, es_lamost, 'SDSS', k_sdss, k_lamost, sigma_clip)

        logger.info(f'Iteration {i}. SDSS scaling = {round(k_sdss, 3)}. LAMOST scaling = {round(k_lamost, 3)}')
        
        if (is_lamost_convergent) and (is_sdss_convergent):
            logger.info('Convergence is reached for both error scalings.')
            break
    else:
        logger.info('Maximum number of iterations reached')        
    return k_sdss, k_lamost

def get_error_scaling_lamost_only(df_repeat, sigma_clip=5.0, max_iter=10):
    '''
    A function to obtain the error scaling
    '''
    def update_error_scaling(s_var, es_var, s_fiducial, es_fiducial, survey, k=1.0, sigma_clip=3.0, convergence_tol=0.005):
        '''
        A function to calculate the error scaling for SDSS and LAMOST.
        '''
        try:
            logger.info(f'Updating error scaling for LAMOST using {survey} as the fiducial.')
            # Scale the errors
            es_var_scaled = k * es_var

            # Calculate the pairwise statistics
            epsilon = (s_var - s_fiducial) / np.sqrt(es_var_scaled**2 + es_fiducial**2)

            # Apply sigma clipping before calculating the new error scaling
            logger.info(f'Applying {sigma_clip} sigma clipping...')
            sigma_clip_filter = np.logical_and(~np.isnan(epsilon), np.absolute(epsilon) < sigma_clip)
            es_var_clipped = es_var_scaled[sigma_clip_filter]
            es_fiducial_clipped = es_fiducial[sigma_clip_filter]
            epsilon_clipped = epsilon[sigma_clip_filter]
            logger.info(f'Number of comparisons remaining = {len(epsilon_clipped)}.')

            # Update the error scaling and check convergence
            N = len(epsilon_clipped)
            rms_var = (1 / N) * np.sum(es_var_clipped**2)
            rms_fiducial = (1 / N) * np.sum(es_fiducial_clipped**2)
            f2 = np.std(epsilon_clipped)**2
            k_new = np.sqrt(f2 + (f2 - 1) * (rms_fiducial / rms_var))
            k_updated = k * k_new
            is_convergent = np.absolute((k_updated - k) / k_updated) * 100 < convergence_tol

            logger.info(f'New scaling for LAMOST using {survey} fiducial = {k_updated}.')
            return k_updated, is_convergent
        except Exception as e:
            logger.info(f'Finding scaling for LAMOST using {survey} fiducial failed. Reason: {e}')
            return k, True

    # SDSS and LAMOST veldisp
    s_6df = df_repeat['s_6df'].to_numpy()
    es_6df = df_repeat['es_6df'].to_numpy()
    s_sdss = df_repeat['s_sdss'].to_numpy()
    es_sdss = df_repeat['es_sdss'].to_numpy()
    s_lamost = df_repeat['s_lamost'].to_numpy()
    es_lamost = df_repeat['es_lamost'].to_numpy()
    
    # Initial scalings for SDSS and LAMOST
    k_lamost_sdss_fid = 1.0
    k_lamost_6df_fid = 1.0
    
    # Find the error scalings
    for i in range(max_iter):
        # Using SDSS fiducial 
        k_lamost_sdss_fid, is_lamost_sdss_fid_convergent = update_error_scaling(s_lamost, es_lamost, s_sdss, es_sdss, 'SDSS', k_lamost_sdss_fid, sigma_clip)
        
        # Using 6dFGS fiducial
        k_lamost_6df_fid, is_lamost_6df_fid_convergent = update_error_scaling(s_lamost, es_lamost, s_6df, es_6df, '6dFGS', k_lamost_6df_fid, sigma_clip)

        logger.info(f'Iteration {i}. LAMOST scaling using SDSS fiducial = {round(k_lamost_sdss_fid, 3)}. LAMOST scaling using 6dFGS fiducial = {round(k_lamost_6df_fid, 3)}')
        
        if (is_lamost_sdss_fid_convergent) and (is_lamost_6df_fid_convergent):
            logger.info('Convergence is reached for both error scalings.')
            break
    else:
        logger.info('Maximum number of iterations reached')
        
    return k_lamost_sdss_fid, k_lamost_6df_fid

def get_offset(k_sdss=1.0, k_lamost=1.0, runs=3, cut=0.2, target=0.5, nboot=10, level=0., max_iter=100., random_seed=42):
    '''
    A function to get the offset in log velocity dispersions (or equivalently scaling in linear velocity dispersions).

    Parameters
    ----------
    k_sdss : SDSS error scaling obtained previously.
    k_lamost : LAMOST error scaling obtained previously.
    runs : the number of surveys to be compared.
    cut : during offset calculation, reject galaxies in which the offset is larger than this limit.
    target : stop the iteration when the maximum offset significance of the three surveys falls below this limit.
    nboot : the number of simulations. The first simulation is using the actual data, the rest is using Monte Carlo samples.
    level : minimum significance in which an offset should be applied.
    max_iter : maximum number of iterations to find the offset.
    random_seed : random seed number for reproducibility.
    '''
    try:
        # Set random seed
        np.random.seed(random_seed)

        # Open the data
        df = pd.read_csv(VELDISP_ORI_OUTPUT_FILEPATH)

        # Apply the error scalings
        df['es_sdss'] = df['es_sdss'] * k_sdss
        df['es_lamost'] = df['es_lamost'] * k_lamost

        # Initial sigmas and error of sigmas
        isig = df[['s_6df', 's_sdss', 's_lamost']].to_numpy()
        idsig = df[['es_6df', 'es_sdss', 'es_lamost']].to_numpy()

        # List to store the offset from each simulation
        totoffs = []

        # Iterate for every bootstrap instance (simulated sigmas)
        for boot in range(nboot):
            # For the first simulation, use the measurements directly
            if (boot == 0):
                ssig = isig
                dsig = idsig
            # Else, use Monte Carlo sample
            else:
                ssig = isig + idsig * np.random.normal(size=idsig.shape)
                dsig = idsig
            
            # Reset the total offset at the beginning of each simulation
            totoff = np.zeros(runs)  # Total offset (scaled to whichever survey picked)
            iteration = 0
            maxrat = 999
            
            # levels = np.zeros(shape=(max_iter, 3))
            # Start iterating through each simulation to obtain the offset
            while ((maxrat >= target) and (iteration < max_iter)):
                iteration += 1
                # logger.info(f'================== Simulation {boot}. Iteration {iteration}. Offsets = {totoff} ==================')

                # Apply the offset at the beginning of each iteration
                sig = ssig - totoff
                # Set maximum significance as 0
                maxrat = 0
                # Number of surveys with significant offset (set as 0)
                nbig = 0
                
                # Calculate the offset for each survey
                for j, survey in enumerate(SURVEY_LIST):
                    off = np.zeros(runs)
                    err = np.zeros(runs)
                    norms = np.zeros(runs)
                    rat = np.zeros(runs)
                    
                    # Find the list of galaxies with measurements in the target survey
                    target_survey_filter = ~np.isnan(sig[:, j])
                    # logger.info(f'Number of galaxies in {survey} = {len(target_survey_filter)}.')
                    
                    # Calculate for each galaxy
                    sig_over_dsig = sig / (dsig**2)
                    one_over_dsig = 1 / (dsig**2)
                    count_notnan = (~np.isnan(sig)).astype(int)
                    
                    # Target survey quantities
                    x = sig_over_dsig[target_survey_filter, j]
                    dx = one_over_dsig[target_survey_filter, j]
                    m = count_notnan[target_survey_filter, j]
                    
                    # Other surveys quantities
                    y = np.nansum(np.delete(sig_over_dsig, j, axis=1), axis=1)[target_survey_filter]
                    dy = np.nansum(np.delete(one_over_dsig, j, axis=1), axis=1)[target_survey_filter]
                    n = np.nansum(np.delete(count_notnan, j, axis=1), axis=1)[target_survey_filter]

                    # Calculate for each galaxy again
                    wt = m * n
                    x = x / dx
                    dx = np.sqrt(1 / dx)
                    y = y / dy
                    dy = np.sqrt(1 / dy)
                    diff = x - y
                    
                    # Filter galaxies where diff > cut
                    offset_cut_filter = np.absolute(diff) < cut
                    off[j] = np.sum((wt * diff)[offset_cut_filter])
                    err[j] = np.sum((wt**2 * (dx**2 + dy**2))[offset_cut_filter])
                    norms[j] = np.sum(wt[offset_cut_filter])
                    
                    # Determine the offset
                    off[j] = off[j] / norms[j]
                    err[j] = np.sqrt(err[j]) / norms[j]
                    rat[j] = off[j] / err[j]
                    # logger.info(f'Survey = {survey}. N = {int(norms[j])}. Offset = {round(off[j], 3)}. Error = {round(err[j], 3)}. Level = {round(rat[j], 3)}.')

                    absrat = np.absolute(rat[j])
                    if absrat > maxrat:
                        maxrat = absrat
                    if absrat >= target:
                        nbig += 1
                    if absrat >= level:
                        totoff[j] = totoff[j] + off[j]
                # logger.info(f"There are {nbig} significant surveys.")
            
            # Subtract with the fiducial survey (taken to be SDSS)
            totoff = totoff - totoff[1]
            totoffs.append(totoff - totoff[1])

        # Save the totoffs from all simulations
        df = pd.DataFrame(data=totoffs, columns=['off_6df', 'off_sdss', 'off_lamost'])
        df.to_csv(VELDISP_TOTOFF_OUTPUT_FILEPATH, index=False)
        return df
    except Exception as e:
        logger.error(f"Finding velocity dispersion offsets failed. Reason: {e}")

def get_mean_offset(totoffs, nbins=10):
    '''
    A function to obtain the mean offset, given the offsets from all the simulations.
    '''
    # Define the Gaussian function
    def gaus(x, xmean, sigma):
        y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x - xmean) / sigma)**2)
        return y

    try:
        # Obtain 6dFGS mean offset
        off_6df = totoffs.off_6df.to_numpy()
        y, x_edges = np.histogram(off_6df, bins=10, density=True)
        x = (x_edges[1:] + x_edges[:-1]) / 2
        popt_6df, pcov_6df = curve_fit(gaus, x, y, p0=[np.mean(off_6df), np.std(off_6df)])
        logger.info(f'6dFGS mean offset = {popt_6df[0]}')
        logger.info(f'6dFGS offset standard deviation = {popt_6df[1]}')

        # Obtain LAMOST mean offset
        off_lamost = totoffs.off_lamost.to_numpy()
        y, x_edges = np.histogram(off_lamost, bins=10, density=True)
        x = (x_edges[1:] + x_edges[:-1]) / 2
        popt_lamost, pcov_lamost = curve_fit(gaus, x, y, p0=[np.mean(off_lamost), np.std(off_lamost)])
        logger.info(f'LAMOST mean offset = {popt_lamost[0]}')
        logger.info(f'LAMOST offset standard deviation = {popt_lamost[1]}')

        return popt_6df[0], popt_lamost[0]
    except Exception as e:
        logger.error(f'Finding the mean offsets failed. Reason: {e}.')

def generate_comparison_plot(k_6df=1.0, k_sdss=1.0, k_lamost=1.0, off_6df=0., off_sdss=0., off_lamost=0., sigma_clip=5.0):
    '''
    A function to generate the chi distributions histogram before vs after applying the calibrations.
    '''
    # CONSTANTS
    BIN_LIST = [5, 40, 9]
    XLIM_LIST = [(-6, 6), (-6, 6), (-6, 6)]
    XLABEL_LIST = [r'$\epsilon_\text{6dFGS-SDSS}$', r'$\epsilon_\text{SDSS-LAMOST}$', r'$\epsilon_\text{6dFGS-LAMOST}$']

    logger.info('Generating comparison plot with the following inputs:')
    logger.info(f'k_6df: {k_6df}')
    logger.info(f'k_sdss: {k_sdss}')
    logger.info(f'k_lamost: {k_lamost}')
    logger.info(f'off_6df: {off_6df}')
    logger.info(f'off_sdss: {off_sdss}')
    logger.info(f'off_lamost: {off_lamost}')

    try:
        df = pd.read_csv(VELDISP_ORI_OUTPUT_FILEPATH)

        # Apply the offsets
        df['s_6df_scaled'] = df['s_6df'] - off_6df
        df['es_6df_scaled'] = df['es_6df'] * k_6df
        df['s_sdss_scaled'] = df['s_sdss'] - off_sdss
        df['es_sdss_scaled'] = df['es_sdss'] * k_sdss
        df['s_lamost_scaled'] = df['s_lamost'] - off_lamost
        df['es_lamost_scaled'] = df['es_lamost'] * k_lamost

        # Calculate the epsilons (without offset)
        df['epsilon_6df_sdss'] = (df['s_6df'] - df['s_sdss']) / np.sqrt(df['es_6df']**2 + df['es_sdss']**2)
        df['epsilon_sdss_lamost'] = (df['s_sdss'] - df['s_lamost']) / np.sqrt(df['es_sdss']**2 + df['es_lamost']**2)
        df['epsilon_6df_lamost'] = (df['s_6df'] - df['s_lamost']) / np.sqrt(df['es_6df']**2 + df['es_lamost']**2)
        epsilon = df[['epsilon_6df_sdss', 'epsilon_sdss_lamost', 'epsilon_6df_lamost']]

        # Calculate the epsilons (with offset)
        df['epsilon_6df_sdss_scaled'] = (df['s_6df_scaled'] - df['s_sdss_scaled']) / np.sqrt(df['es_6df_scaled']**2 + df['es_sdss_scaled']**2)
        df['epsilon_sdss_lamost_scaled'] = (df['s_sdss_scaled'] - df['s_lamost_scaled']) / np.sqrt(df['es_sdss_scaled']**2 + df['es_lamost_scaled']**2)
        df['epsilon_6df_lamost_scaled'] = (df['s_6df_scaled'] - df['s_lamost_scaled']) / np.sqrt(df['es_6df_scaled']**2 + df['es_lamost_scaled']**2)
        epsilon_scaled = df[['epsilon_6df_sdss_scaled', 'epsilon_sdss_lamost_scaled', 'epsilon_6df_lamost_scaled']]

        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 5))

        # Plot before and after scaling + offset
        for i, ax in enumerate(axs):
            data = epsilon[epsilon.columns[i]].dropna()
            ax.hist(data, bins=BIN_LIST[i], density=True, alpha=0.5)
            
            data_scaled = epsilon_scaled[epsilon_scaled.columns[i]].dropna()
            ax.hist(data_scaled, bins=BIN_LIST[i], density=True, alpha=0.5)
            
            # Misc
            ax.grid(linestyle=":")
            ax.set_title(f'N = {len(data)}')
            ax.set_xlim(XLIM_LIST[i])
            ax.set_xlabel(XLABEL_LIST[i], fontsize=18)
            ax.set_xticks(ax.get_xticks()[1:-1])
    
            if i==0:
                ax.set_ylabel(r'$N$', fontsize=14)
            
        # Plot standard normal Gaussians (target)
        x = np.arange(start=-10., stop=10., step=0.0001)
        y = norm.pdf(x, loc=0., scale=1.)
        for ax in axs:
            ax.plot(x, y, c='k', lw=1.0)

        plt.subplots_adjust(wspace=0)

        img_output_path = os.path.join(ROOT_PATH, f'img/veldisp_comparison_{sigma_clip}sigma.png')
        logger.info(f'Saving image to {img_output_path}')
        plt.tight_layout()
        fig.savefig(img_output_path, dpi=300)
        return
    except Exception as e:
        logger.error(f'Generating comparison plot failed. Reason: {e}.')

def apply_scalings(error_scalings, offsets):
    '''
    This function applies the error scalings and velocity dispersion offsets to the data.
    '''
    try:
        for survey in SURVEY_LIST:
            df = pd.read_csv(INPUT_FILEPATH[survey])
            df['s_scaled'] = df['s'] - offsets[survey]
            df['es_scaled'] = df['es'] * error_scalings[survey]
            df.to_csv(OUTPUT_FILEPATH[survey], index=False)
    except Exception as e:
        logger.error(f'Applying scalings failed. Reason: {e}.')

def main():
    logger.info(f'Finding repeat measurements...')
    start = time.time()
    df = get_common_galaxies()
    end = time.time()
    logger.info(f'Finding repeat measurements successful! Time elapsed = {round(end - start, 2)} s.')

    # logger.info(f'Finding error scalings using old method (vary k_lamost and k_sdss)')
    # start = time.time()
    # k_sdss, k_lamost = get_error_scaling_old_method(df, sigma_clip=3.5)
    # end = time.time()
    # logger.info(f'Finding error scalings using old method successful! Time elapsed = {round(end - start, 2)} s.')

    logger.info(f'Finding LAMOST error scaling by comparing with SDSS and 6dFGS...')
    start = time.time()
    k_lamost_sdss_fid, k_lamost_6df_fid = get_error_scaling_lamost_only(df, sigma_clip=3.5)
    end = time.time()
    logger.info(f'Finding LAMOST error scaling by comparing with SDSS and 6dFGS successful! Time elapsed = {round(end - start, 2)} s.')

    # Set the errors
    k_6df = 1.0
    k_sdss = 1.0
    k_lamost = k_lamost_sdss_fid

    logger.info(f"Finding the velocity dispersion offset...")
    totoffs = get_offset(k_sdss, k_lamost, nboot=100)
    off_6df = totoffs.loc[0, ['off_6df']].values[0]
    off_lamost = totoffs.loc[0, ['off_lamost']].values[0]

    logger.info(f"Generating the epsilon comparison plot...")
    generate_comparison_plot(k_sdss=k_sdss, k_lamost=k_lamost, off_6df=off_6df, off_lamost=off_lamost)
    
    logger.info(f"Applying the scalings...")
    off_sdss = 0.0
    error_scalings = {
        '6dFGS': k_6df,
        'SDSS': k_sdss,
        'LAMOST': k_lamost
    }
    offsets = {
        '6dFGS': off_6df,
        'SDSS': off_sdss,
        'LAMOST': off_lamost
    }
    apply_scalings(error_scalings, offsets)

if __name__ == '__main__':
    main()