import os
import time
import numpy as np
import pandas as pd
from utils.constants import *
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('zms_cut')

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/lamost.csv')
}

HIGH_Z_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/zms_cut/lamost.csv')
}

LOW_Z_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/lamost.csv')
}

# Grab 6dFGS offset
totoff = pd.read_csv(os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/totoffs.csv'))
off_6df = totoff.loc[0, ['off_6df']].values[0]

# Selection criteria constants
UPPER_Z_LIMIT = 16120.0 / LIGHTSPEED
LOWER_Z_LIMIT = 3000.0 / LIGHTSPEED
UPPER_MAG_LIMIT = 13.65
LOWER_VELDISP_LIMIT = np.log10(112) - off_6df

def apply_selection():
    '''
    A function to apply redshift, magnitude, and velocity dispersions cut to the veldisp-calibrated sample.
    '''
    for survey in SURVEY_LIST:
        logger.info(f"{'=' * 25} {survey} {'=' * 25}")

        df = pd.read_csv(INPUT_FILEPATH[survey])
        old_count = len(df)
        logger.info(f"Original number of galaxies in {survey}: {old_count}")
        
        # Apply upper CMB redshift limit
        df = df[df['z_dist_est'] <= UPPER_Z_LIMIT]
        new_count = len(df)
        logger.info(f"Number of galaxies after cz <= 16120 = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count
        
        # Apply upper magnitude limit
        df = df[(df['j_m_ext'] - df['extinction_j']) <= UPPER_MAG_LIMIT]
        new_count = len(df)
        logger.info(f"Number of galaxies after (m_j - extinction_j) <= 13.65 = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count
        
        # Apply upper veldisp limit
        df = df[df['s_scaled'] >= LOWER_VELDISP_LIMIT]
        new_count = len(df)
        logger.info(f"Number of galaxies after s_scaled >= log10(112) - 6dFGS_offset ({round(off_6df, 3)}) = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count
        
        # Apply lower CMB redshift limit
        df_high_z = df[df['z_dist_est'] >= LOWER_Z_LIMIT]
        new_count = len(df_high_z)
        logger.info(f"Number of galaxies after cz >= 3000 = {new_count} | Discarded galaxies = {old_count - new_count}")
        
        # Save the remaining high-redshift galaxies
        df_high_z.to_csv(HIGH_Z_OUTPUT_FILEPATH[survey], index=False)
        
        # Save the low-redshift galaxies (will not be used to fit the FP due to high scatter but PVs will still be measured)
        df_low_z = df[df['z_dist_est'] <= LOWER_Z_LIMIT]
        logger.info(f"Number of galaxies with cz <= 3000 = {len(df_low_z)}")
        df_low_z.to_csv(LOW_Z_OUTPUT_FILEPATH[survey], index=False)

        logger.info('\n')

def main():
    try:
        logger.info('Applying selection criteria...')
        start = time.time()
        apply_selection()
        end = time.time()
        logger.info(f'Applying selection criteria successful! Time elapsed: {round(end - start, 2)} s.')
        logger.info('\n')
    except Exception as e:
        logger.error(f'Applying selection criteria failed. Reason: {e}.')

if __name__ == '__main__':
    main()