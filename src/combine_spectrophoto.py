import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('get_spectrophoto')

# 6dFGS paths
SDFGS_FP_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_fp_vizier.fits')
SDFGS_TMASS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/2mass/sdfgs_tmass.fits')
SDFGS_VELDISP_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_veldisp_vizier.fits')
SDFGS_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/6dfgs.csv')

# SDSS paths
SDSS_SPECTRO_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/sdss/SDSS_spectro.csv')
LAMOST_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')

def combine_6df_spectrophoto():
    '''
    A function to combine the 6dFGS FP galaxies with 2MASS photometry and velocity dispersions.
    Steps:
    1. Open the 6dFGS FP sample (contains the FP galaxies, radii, and group information)
    2. Open the 2MASS data (contains the galactic coordinates, axial ratios, radii, magnitudes, and mean surface brightnesses)
    3. Merge FP-2MASS dataframes by 2MASS id by indices
    4. Perform sanity check to see if the data are consistent (compare RAJ2000 and DEJ2000 with ra_01 and dec_01)
    5. Open the velocity dispersions table
    6. Select the measurement with the best S_N by doing the following: sort by S_N and drop duplicates by 2MASS id
    7. Merge FP-2MASS-veldisp
    8. Save the table to data/preprocessed/spectrophoto
    '''

    try:
        # Vizier first table (FP sample)
        logger.info('Opening the 6dFGS FP sample...')
        req_cols = ['_2MASX', 'Jlogr', 'n', 'Hlogr', 'Klogr', '_6dFGS', 'RAJ2000', 
                    'DEJ2000', 'cz', 'Mt', 'Group', 'Nr', 'Ng', 'cz_gr']
        with fits.open(SDFGS_FP_FILEPATH) as hdul:
            df_fp = Table(hdul[1].data).to_pandas()[req_cols]
            df_fp['RAJ2000'] *= 15

        # Open the 2MASS data
        logger.info('Opening the 2MASS photometry data...')
        req_cols = ['ra_01', 'dec_01', 'glon', 'glat', 'j_ba', 'h_ba', 'k_ba', 
                    'sup_ba', 'r_ext', 'j_m_ext', 'h_m_ext', 'k_m_ext', 'j_r_eff', 'j_mnsurfb_eff', 
                    'h_r_eff', 'h_mnsurfb_eff', 'k_r_eff', 'k_mnsurfb_eff']
        df_2mass = pd.read_csv('data/raw/2mass/sdfgs_tmass.csv')[req_cols]

        # Merge FP + 2MASS
        logger.info('Merging 6dFGS FP data with 2MASS data...')
        df = df_fp.merge(df_2mass, left_index=True, right_index=True)

        # Sanity test (check RAJ2000 DEJ2000 vs ra_01 dec_01)
        max_delta_ra = np.absolute(max(df['RAJ2000'] - df['ra_01']))
        max_delta_dec = np.absolute(max(df['DEJ2000'] - df['dec_01']))
        tol_ = 0.001
        logger.info(f'Max delta RA: {max_delta_ra}')
        logger.info(f'Max delta DEC: {max_delta_dec}')

        if (max_delta_ra > tol_) or (max_delta_dec > tol_):
            logger.info('The two tables do not match.')
            raise
        else:
            logger.info('The coordinates in 6dFGS FP and 2MASS are consistent.')
            df = df.drop(['ra_01', 'dec_01'], axis=1)

        # Vizier second table (veldisp data)
        logger.info('Opening the 6dFGS veldisp data...')
        req_cols = ['_2MASX', 'MJD', 'z', 'S_N', 'Vd', 'e_Vd']
        with fits.open('data/raw/6dfgs/sdfgs_veldisp_vizier.fits') as hdul:
            df_veldisp = Table(hdul[1].data).to_pandas()[req_cols]
            
        ## Drop duplicated rows (select the one with the highest S_N)
        df_veldisp = df_veldisp.sort_values(by='S_N', ascending=False)
        df_veldisp = df_veldisp.drop_duplicates(subset='_2MASX')

        # Merge the with the velocity dispersion data
        logger.info('Merging 6dFGS FP+2MASS with 6dFGS veldisp...')
        df = df.merge(df_veldisp, on='_2MASX')

        # Change the primary key nane
        df = df.rename({'_2MASX': 'tmass'}, axis=1)
        df['tmass'] = '2MASX' + df['tmass']

        # Save the resulting table
        logger.info(f'Number of galaxies = {len(df)}. Saving the table to {SDFGS_OUTPUT_FILEPATH}.')
        df.to_csv(SDFGS_OUTPUT_FILEPATH, index=False)

        return
    except Exception as e:
        logger.error(f"Combining 6dFGS spectroscopy and photometry data failed. Reason: {e}")

if __name__ == '__main__':
    combine_6df_spectrophoto()
