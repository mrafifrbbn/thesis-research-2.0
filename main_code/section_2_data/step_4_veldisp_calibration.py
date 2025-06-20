import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.optimize import curve_fit
from scipy.stats import norm
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv(override=True)

from utils.constants import *

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('veldisp_calibration')

# Set random seed
np.random.seed(42)

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
VELDISP_TOTOFF_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/totoffs.csv')
VELDISP_SCALED_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/repeat_scaled.csv')


# --------------------------- VARIABLE THAT NEEDS TO BE ADJUSTED --------------------------- #
ERROR_SCALING_METHODS = ['old_method', 'sdss_fiducial', 'lamost_only', 'sdss_only']
METHOD_NO: int = 2       # Fill 0, 1, 2, or 3

def get_common_galaxies() -> pd.DataFrame:
    '''
    A function to find galaxies with repeat velocity dispersion measurements.
    '''
    try:
        # Grab all of the data and only use their 2MASS and survey ID's, also grab the velocity dispersion measurements
        logger.info('Opening r, s, i data...')
        data_6df = pd.read_csv(INPUT_FILEPATH['6dFGS'])[['tmass', '_6dFGS', 's', 'es']]
        data_6df_id = data_6df[['tmass','_6dFGS']]
        data_6df_veldisp = data_6df[['tmass', 's', 'es']].rename(columns={'s': 's_6df', 'es': 'es_6df'})

        data_sdss = pd.read_csv(INPUT_FILEPATH['SDSS'])[['tmass', 'objid', 's', 'es']]
        data_sdss_id = data_sdss[['tmass','objid']]
        data_sdss_veldisp = data_sdss[['tmass', 's', 'es']].rename(columns={'s': 's_sdss', 'es': 'es_sdss'})

        data_lamost = pd.read_csv(INPUT_FILEPATH['LAMOST'])[['tmass', 'obsid', 's', 'es']]
        data_lamost_id = data_lamost[['tmass','obsid']]
        data_lamost_veldisp = data_lamost[['tmass', 's', 'es']].rename(columns={'s': 's_lamost', 'es': 'es_lamost'})

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


def calculate_sdss_lamost_scaling(df: pd.DataFrame):

    df_full = df.copy()
    df_full = df_full[(~df_full['s_sdss'].isna()) & (~df_full['s_lamost'].isna())]

    k_best_list = []
    chisq_upper_list = []
    chisq_lower_list = []
    for boot in range(5000):

        df_ = df_full.copy()

        # Unpack data
        s_sdss = df_['s_sdss'].to_numpy()
        es_sdss = df_['es_sdss'].to_numpy()
        s_lamost = df_['s_lamost'].to_numpy()
        es_lamost = df_['es_lamost'].to_numpy()

        # For the first simulation, use the measurements directly
        if (boot > 0):
            s_sdss = s_sdss + es_sdss * np.random.normal(size=es_sdss.shape)
            s_lamost = s_lamost + es_lamost * np.random.normal(size=es_lamost.shape)

        # Calculate chi-squared
        chisq = (s_sdss - s_lamost) / (np.sqrt(es_sdss**2 + es_lamost**2))

        # Calculate std of chi-squared (equal to the scaling constant that makes std=1)
        chisq_std = np.std(chisq)

        # Save all results
        k_best_list.append(chisq_std)

    # Scaling and its error estimate
    k_best = k_best_list[0]
    k_best_err = np.std(k_best_list)

    return k_best, k_best_err


def get_offset(k_6df: float = 1.0, k_sdss: float = 1.0, k_lamost: float = 1.0, runs: int = 3, cut: float = 0.2, target: float = 0.5, nboot: int = 10, level: float = 0.0, max_iter: int = 100., random_seed: int = 42) -> pd.DataFrame:
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
        df['es_6df'] = df['es_6df'] * k_6df
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

def get_mean_offset(totoffs: pd.DataFrame, nbins: int = 10) -> Tuple[float, float]:
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


def apply_scalings(error_scalings: Dict[str, float], offsets: Dict[str, float]) -> None:
    '''
    This function applies the error scalings and velocity dispersion offsets to the data.
    '''
    for survey in SURVEY_LIST:
        df = pd.read_csv(INPUT_FILEPATH[survey])
        df['s_scaled'] = df['s'] - offsets[survey]
        df['es_scaled'] = df['es'] * error_scalings[survey]
        df.to_csv(OUTPUT_FILEPATH[survey], index=False)


def main():
    try:
        logger.info(f'Finding repeat measurements...')
        start = time.time()
        df = get_common_galaxies()
        end = time.time()
        logger.info(f'Finding repeat measurements successful! Time elapsed = {round(end - start, 2)} s.')

        logger.info("Calculating the common scaling for SDSS and LAMOST...")
        k, k_err = calculate_sdss_lamost_scaling(df)
        logger.info(f"SDSS and LAMOST common errors: {k} ± {k_err}")

        # Set the errors
        k_6df = 1.0
        k_6df_err = 0.0
        k_sdss = k
        k_sdss_err = k_err
        k_lamost = k
        k_lamost_err = k_err

        # Save the scalings
        scalings = np.array([[k_6df, k_6df_err, k_sdss, k_sdss_err, k_lamost, k_lamost_err]])
        VELDISP_SCALING_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/veldisp_calibration/scaling_sdss_lamost.csv')
        pd.DataFrame(data=scalings, columns=['k_6df', 'k_6df_err', 'k_sdss', 'k_sdss_err', 'k_lamost', 'k_lamost_err']).to_csv(VELDISP_SCALING_OUTPUT_FILEPATH, index=False)

        # Calculate the error-weighted offsets
        logger.info(f"Finding the velocity dispersion offset...")
        totoffs = get_offset(k_6df, k_sdss, k_lamost, nboot=100)
        off_6df = totoffs.loc[0, ['off_6df']].values[0]
        off_sdss = totoffs.loc[0, ['off_sdss']].values[0]
        off_lamost = totoffs.loc[0, ['off_lamost']].values[0]
        
        logger.info(f"Applying the scalings...")
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
        logger.info(f"Applying scalings. Error scaling: {error_scalings} | offsets: {offsets}")
        apply_scalings(error_scalings, offsets)

        logger.info("Velocity dispersion calibration successful!")
    except Exception as e:
        logger.error("Velocity dispersion calibration failed.", exc_info=True)

if __name__ == '__main__':
    main()