import os
import time
import numpy as np
import pandas as pd
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('veldisp_calibration')

INPUT_FILEPATH = {
    '6dFGS': 'data/processed/rsi_derived/6dfgs.csv',
    'SDSS': 'data/processed/rsi_derived/sdss.csv',
    'LAMOST': 'data/processed/rsi_derived/lamost.csv'
}

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

        data_sdss = pd.read_csv(INPUT_FILEPATH['SDSS'])[['tmass', 'objID', 's', 'es']]
        data_sdss_id = data_sdss[['tmass','objID']]
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
                            .drop('objID_y',axis=1)\
                            .rename(columns={'objID_x':'objID'})
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
        repeat_6df_lamost_only.loc[:, ['objID']] = np.nan
        repeat_sdss_lamost_only.loc[:, ['_6dFGS']] = np.nan

        # Concatenate the dataframes
        df = pd.concat([repeat_3, repeat_6df_sdss_only, repeat_6df_lamost_only, repeat_sdss_lamost_only])
        # Join with the velocity dispersion measurements
        df = df.merge(data_6df_veldisp, how='left', on='tmass')\
            .merge(data_sdss_veldisp, how='left', on='tmass')\
            .merge(data_lamost_veldisp, how='left', on='tmass')
        logger.info(f'Number of unique common galaxies in 6dFGS-SDSS-LAMOST = {len(df)}')

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
        logger.info(f'Applying sigma clipping...')
        sigma_clip_filter = np.logical_and(~np.isnan(epsilon), np.absolute(epsilon) < sigma_clip)
        es_sdss_clipped = es_sdss_scaled[sigma_clip_filter]
        es_lamost_clipped = es_lamost_scaled[sigma_clip_filter]
        epsilon_clipped = epsilon[sigma_clip_filter]
        logger.info(f'Number of comparisons remaining = {len(epsilon_clipped)}.')
        
        # Update the error scaling and check convergence
        N = len(epsilon_clipped)
        r12 = (1 / N) * np.sum(es_sdss_clipped**2)
        r32 = (1 / N) * np.sum(es_lamost_clipped**2)
        f2 = np.std(epsilon_clipped)**2
        
        if survey == 'LAMOST':
            k_new = np.sqrt(f2 + (f2 - 1)*(r12 / r32))
            k_updated = k_lamost * k_new
            is_convergent = np.absolute((k_updated - k_lamost) / k_updated) * 100 < convergence_tol
        elif survey == 'SDSS':
            k_new = np.sqrt(f2 + (f2 - 1) * (r32 / r12))
            k_updated = k_sdss * k_new
            is_convergent = np.absolute((k_updated - k_sdss) / k_updated) * 100 < convergence_tol
        
        logger.info(f'New scaling for {survey} = {k_updated}.')
        return k_updated, is_convergent
    except Exception as e:
        logger.error(f'Updating scaling for {survey} failed. Reason: {e}.')

def main():
    logger.info(f'Finding repeat measurements...')
    start = time.time()
    df = get_common_galaxies()
    end = time.time()
    logger.info(f'Finding repeat measurements successful! Time elapsed = {round(end - start, 2)} s.')

    # Constants
    offset_threshold = 0.2      # Reject measurements that are too different
    sigma_clip = 3.0            # Sigma clipping threshold
    k_sdss = 1.0                # Initial scaling for SDSS 
    k_lamost = 1.0              # Initial scaling for LAMOST
    Nmax = 10                   # Maximum iteration

    # SDSS and LAMOST veldisp
    df_calib = df[(df['objID'].notna()) 
                    & (df['obsid'].notna()) 
                    & (np.absolute(df['s_sdss'] - df['s_lamost']) < offset_threshold)]
    s_sdss = df_calib['s_sdss'].to_numpy()
    es_sdss = df_calib['es_sdss'].to_numpy()
    s_lamost = df_calib['s_lamost'].to_numpy()
    es_lamost = df_calib['es_lamost'].to_numpy()

    # Find the error scalings
    for i in range(Nmax):
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

    logger.info(f"{'='*50}")
    logger.info('Final SDSS scaling = %.3f' % k_sdss)
    logger.info('Final LAMOST scaling = %.3f' % k_lamost)
    
if __name__ == '__main__':
    main()