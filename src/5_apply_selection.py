import os
import time
import numpy as np
import pandas as pd

from utils.constants import *
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))

# Create logging instance
logger = get_logger('zms_cut')

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/veldisp_calibrated/lamost.csv')
}

LOW_Z_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/zms_cut/low_z/lamost.csv')
}
create_parent_folder(LOW_Z_OUTPUT_FILEPATH)

HIGH_Z_OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/lamost.csv')
}
create_parent_folder(HIGH_Z_OUTPUT_FILEPATH)

LAMOST_GOOD_PV_LIST_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/r_e_jrl/lamost_good_pv_list.csv')

# Selection criteria constants
UPPER_Z_LIMIT = 16120.0 / LIGHTSPEED
LOWER_Z_LIMIT = 3000.0 / LIGHTSPEED
UPPER_MAG_LIMIT = 13.65

def apply_selection() -> None:
    '''
    A function to apply redshift, magnitude, and velocity dispersions cut to the veldisp-calibrated sample.
    '''
    for survey in SURVEY_LIST:
        logger.info(f"{'=' * 25} {survey} {'=' * 25}")

        df = pd.read_csv(INPUT_FILEPATH[survey])
        old_count = len(df)
        logger.info(f"Original number of galaxies in {survey}: {old_count}")

        # # Bypass all of the selection for 6dFGS (keep all of Christina's galaxies)
        # if (survey == '6dFGS') and not USE_6dFGS_OFFSET:
        #     logger.info(f"Keeping all 6dFGS galaxies...")
        #     df.to_csv(HIGH_Z_OUTPUT_FILEPATH[survey], index=False)
        #     continue

        # 1. Apply upper CMB redshift limit
        df = df[df['z_dist_est'] <= UPPER_Z_LIMIT]
        new_count = len(df)
        logger.info(f"Number of galaxies after cz <= 16120 = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count

        # 2. Apply lower CMB redshift limit
        df_high_z = df[df['z_dist_est'] >= LOWER_Z_LIMIT]
        new_count = len(df_high_z)
        logger.info(f"Number of galaxies after cz >= 3000 = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count

        # 3. Apply upper magnitude limit
        df_high_z = df_high_z[(df_high_z['j_m_ext'] - df_high_z['extinction_j']) <= UPPER_MAG_LIMIT]
        new_count = len(df_high_z)
        logger.info(f"Number of galaxies after (m_j - extinction_j) <= 13.65 = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count

        # 4. Apply lower veldisp limit
        df_high_z = df_high_z[df_high_z['s_scaled'] >= SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]]
        new_count = len(df_high_z)
        logger.info(f"Number of galaxies after s_scaled >= {SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey]} = {new_count} | Discarded galaxies = {old_count - new_count}")
        old_count = new_count

        # 5. Select the low-redshift galaxies (will not be used to fit the FP due to high scatter but PVs will still be measured)
        df_low_z = df[(df['z_dist_est'] <= LOWER_Z_LIMIT) & ((df['j_m_ext'] - df['extinction_j']) <= UPPER_MAG_LIMIT) & (df['s_scaled'] >= SURVEY_VELDISP_LIMIT[SMIN_SETTING][survey])]

        # 6. For LAMOST, select galaxies classified as ETG from John's visual inspections
        if survey == 'LAMOST':
            good_pv_list = pd.read_csv(LAMOST_GOOD_PV_LIST_FILEPATH)[['col1']]
            df_high_z = df_high_z.merge(good_pv_list, left_on='tmass', right_on='col1', how='inner').drop(['col1'], axis=1)
            logger.info(f"Number of LAMOST galaxies after visual inspection = {len(df_high_z)} | Discarded galaxies = {old_count - len(df_high_z)}")
            df_low_z = df_low_z.merge(good_pv_list, left_on='tmass', right_on='col1', how='inner').drop(['col1'], axis=1)
            
        # Save the remaining high-redshift galaxies
        logger.info(f"Final number of galaxies for {survey} = {len(df_high_z)}")
        df_high_z.to_csv(HIGH_Z_OUTPUT_FILEPATH[survey], index=False)

        # Save the low-redshift galaxies
        logger.info(f"Number of galaxies with cz <= 3000 = {len(df_low_z)}")
        df_low_z.to_csv(LOW_Z_OUTPUT_FILEPATH[survey], index=False)
        logger.info('\n')

def main() -> None:
    try:
        logger.info(f"{'=' * 50}")
        logger.info(f'Applying selection criteria using SMIN_SETTING {SMIN_SETTING}...')
        logger.info(SURVEY_VELDISP_LIMIT[SMIN_SETTING])
        start = time.time()
        apply_selection()
        end = time.time()
        logger.info(f'Applying selection criteria successful! Time elapsed: {round(end - start, 2)} s.')
        logger.info('\n')
    except Exception as e:
        logger.error(f'Applying selection criteria failed. Reason: {e}.')

if __name__ == '__main__':
    main()