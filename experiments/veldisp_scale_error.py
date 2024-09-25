import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from scipy.optimize import curve_fit
from scipy.stats import norm
from dotenv import load_dotenv
load_dotenv(override=True)

src_dir = '/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/src'
if not src_dir in sys.path: sys.path.append(src_dir)
utils_dir = '/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/src/utils'
if not utils_dir in sys.path: sys.path.append(utils_dir)
from constants import * # type: ignore
from CosmoFunc import * # type: ignore
from logging_config import get_logger # type: ignore

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import erf

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
VELDISP_TOTOFF_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/totoffs.csv')
VELDISP_SCALED_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/repeat_scaled.csv')

# --------------------------- VARIABLE THAT NEEDS TO BE ADJUSTED --------------------------- #
ERROR_SCALING_METHODS = ['old_method', 'sdss_fiducial', 'lamost_only', 'sdss_only']
METHOD_NO: int = 2       # Fill 0, 1, 2, or 3

def get_common_galaxies() -> None:
    '''
    A function to find galaxies with repeat velocity dispersion measurements.
    '''
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

# --------- Find the error scaling only for LAMOST (assume k_sdss = k_6df = 1) ---------- #
def get_error_scaling_lamost_only(df_repeat: pd.DataFrame, sigma_clip: float = 5.0, max_iter: int = 10) -> Tuple[float, float]:
    '''
    A function to obtain the error scaling
    '''
    def update_error_scaling(s_var: np.ndarray, es_var: np.ndarray, s_fiducial: np.ndarray, es_fiducial: np.ndarray, survey: str, k: float = 1.0, sigma_clip: float = 3.0, convergence_tol: float = 0.005) -> Tuple[float, bool]:
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

    k_lamost_from_sdss_bootstrap = []
    k_lamost_from_6dfgs_bootstrap = []
    for boot in range(500):
        if boot == 0:
            # SDSS and LAMOST veldisp
            s_6df = df_repeat['s_6df'].to_numpy()
            es_6df = df_repeat['es_6df'].to_numpy()
            s_sdss = df_repeat['s_sdss'].to_numpy()
            es_sdss = df_repeat['es_sdss'].to_numpy()
            s_lamost = df_repeat['s_lamost'].to_numpy()
            es_lamost = df_repeat['es_lamost'].to_numpy()
        else:
            # SDSS and LAMOST bootstrap (Monte Carlo) veldisp
            es_6df = df_repeat['es_6df'].to_numpy()
            s_6df = df_repeat['s_6df'].to_numpy() + es_6df * np.random.normal(size=es_6df.shape)

            es_sdss = df_repeat['es_sdss'].to_numpy()
            s_sdss = df_repeat['s_sdss'].to_numpy() + es_sdss * np.random.normal(size=es_sdss.shape)
            
            es_lamost = df_repeat['es_lamost'].to_numpy()
            s_lamost = df_repeat['s_lamost'].to_numpy() + es_lamost * np.random.normal(size=es_lamost.shape)
            
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

        k_lamost_from_sdss_bootstrap.append(k_lamost_sdss_fid)
        k_lamost_from_6dfgs_bootstrap.append(k_lamost_6df_fid)
    return k_lamost_from_sdss_bootstrap, k_lamost_from_6dfgs_bootstrap

def main() -> None:
    logger.info(f'Finding repeat measurements...')
    start = time.time()
    df = get_common_galaxies()
    end = time.time()
    logger.info(f'Finding repeat measurements successful! Time elapsed = {round(end - start, 2)} s.')

    # Set the errors
    k_6df = 1.0
    k_sdss = 1.0
    k_lamost = 1.0

    method = ERROR_SCALING_METHODS[METHOD_NO]
    if method == 'lamost_only':
        logger.info(f'Finding LAMOST error scaling by comparing with SDSS and 6dFGS...')
        start = time.time()
        k_lamost_sdss_fid, k_lamost_6df_fid = get_error_scaling_lamost_only(df, sigma_clip=3.5)
        end = time.time()
        logger.info(f'Finding LAMOST error scaling by comparing with SDSS and 6dFGS successful! Time elapsed = {round(end - start, 2)} s.')

    k_lamost_sdss_fid = np.array(k_lamost_sdss_fid)
    k_lamost_6df_fid = np.array(k_lamost_6df_fid)

    k_lamost_sdss_fid_filepath = os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/k_lamost_sdss_fid.npy')
    np.save(k_lamost_sdss_fid_filepath, k_lamost_sdss_fid)

    k_lamost_6df_fid_filepath = os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/k_lamost_6df_fid.npy')
    np.save(k_lamost_6df_fid_filepath, k_lamost_6df_fid)

if __name__ == '__main__':
    main()