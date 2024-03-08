import os
import time
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('get_spectrophoto')

# 6dFGS paths
SDFGS_FP_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_fp_vizier.fits')
SDFGS_TMASS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/2mass/sdfgs_tmass.csv')
SDFGS_VELDISP_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_veldisp_vizier.fits')
SDFGS_MAGOULAS_FP_SAMPLE = 'data/external/ETG_sample/fp_magoulas.txt'
SDFGS_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/6dfgs.csv')

# SDSS and LAMOST settings and paths
SDSS_LAMOST_SPECTRO_REQ_COLS = {
    'SDSS': ['objid', 'specObjId', 'RA', 'Dec', 'zhelio', 'zhelioerr', 'SIGMA_STARS', 'SIGMA_STARS_ERR'],
    'LAMOST': ['obsid', 'ra', 'dec', 'z_lamost', 'veldisp', 'veldisp_err']
}
SDSS_LAMOST_SPECTRO_FILEPATH = {
    'SDSS': os.path.join(ROOT_PATH, 'data/raw/sdss/sdss_howlett2022.dat'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')
    }
SDSS_LAMOST_TMASS_FILEPATH = {
    'SDSS': os.path.join(ROOT_PATH, 'data/raw/2mass/sdss_tmass.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/raw/2mass/lamost_tmass.csv')
}
SDSS_LAMOST_OUTPUT_FILEPATH = {
    'SDSS': os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/lamost.csv')
}

# Supplementary data paths
JRL_PHOTO_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/r_e_jrl/jhk_r_e.csv')
TEMPEL_GAL_FILEPATH = os.path.join(ROOT_PATH, 'data/external/tempel_group_sdss8/tempel_dr8gal.fits')
TEMPEL_GROUP_FILEPATH = os.path.join(ROOT_PATH, 'data/external/tempel_group_sdss8/tempel_dr8gr.fits')

def combine_6df_spectrophoto():
    '''
    A function to combine the 6dFGS FP galaxies with 2MASS photometry and velocity dispersions.
    Steps:
    1. Open the 6dFGS FP sample (contains the FP galaxies, radii, and group information)
    2. Open the 2MASS data (contains the galactic coordinates, axial ratios, radii, magnitudes, and mean surface brightnesses)
    3. Join FP-2MASS dataframes by indices
    4. Perform sanity check to see if the data are consistent (compare RAJ2000 and DEJ2000 with ra_01 and dec_01)
    5. Open the velocity dispersions table
    6. Select the measurement with the best S_N by doing the following: sort by S_N and drop duplicates by 2MASS id
    7. Join 6dFGS FP-2MASS-veldisp
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
        req_cols = ['ra_01', 'dec_01', 'designation', 'glon', 'glat', 'j_ba', 'h_ba', 'k_ba', 
                    'sup_ba', 'r_ext', 'j_m_ext', 'h_m_ext', 'k_m_ext', 'j_r_eff', 'h_r_eff', 'k_r_eff']
        df_2mass = pd.read_csv(SDFGS_TMASS_FILEPATH)[req_cols]

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
        with fits.open(SDFGS_VELDISP_FILEPATH) as hdul:
            df_veldisp = Table(hdul[1].data).to_pandas()[req_cols]

        ## Drop duplicated rows (select the one with the highest S_N)
        df_veldisp = df_veldisp.sort_values(by='S_N', ascending=False)
        df_veldisp = df_veldisp.drop_duplicates(subset='_2MASX')

        # Merge the with the velocity dispersion data
        logger.info('Merging the 6dFGS veldisp data...')
        df = df.merge(df_veldisp, on='_2MASX')

        # Select Christina's FP sample (8803 galaxies)
        logger.info("Opening Christina's FP sample and performing inner join...")
        df_magoulas = pd.read_csv(SDFGS_MAGOULAS_FP_SAMPLE, delim_whitespace=True)[['#6dFGS_ID']]
        df_magoulas = df_magoulas.rename({'#6dFGS_ID': '_6dFGS'}, axis=1)
        df = df.merge(df_magoulas, on='_6dFGS', how='inner')

        # Add prefix '2MASX' to 2MASS id value
        df['_2MASX'] = '2MASX' + df['_2MASX']
        df = df.rename({'_2MASX': 'tmass'}, axis=1)

        # Save the resulting table
        logger.info(f"Number of galaxies = {len(df)}. Number of unique galaxies = {df.designation.nunique()}. Saving the table to {SDFGS_OUTPUT_FILEPATH}.")
        df.to_csv(SDFGS_OUTPUT_FILEPATH, index=False)

        return
    except Exception as e:
        logger.error(f"Combining 6dFGS spectroscopy and photometry data failed. Reason: {e}")

def combine_sdss_lamost_spectrophoto():
    '''
    A function to combine the SDSS and LAMOST spectroscopy (veldisp), 2MASS photometry, John's GALFIT measurements, 
    and Tempel's cluster data.
    Steps:
    1. Open the SDSS/LAMOST spectroscopy data (contains redshift and veldisp)
    2. Open the 2MASS data (contains the galactic coordinates, axial ratios, radii, magnitudes, and mean surface brightnesses)
    3. Join SDSS/LAMOST-2MASS dataframes by indices
    4. Perform sanity check to see if the data are consistent (compare ra and dec with ra_01 and dec_01)
    5. Open John's GALFIT measurements (radii, PSF corrections, other criteria)
    6. Join SDSS/LAMOST-2MASS-JRL on 2MASS id
    7. Open Tempel's individual galaxies and clusters data (join both of them first to get each galaxy's cluster redshift)
    8. Join SDSS/LAMOST-2MASS-JRL-Tempel on galaxy's ra and dec (Tempel does not provide SDSS objID)
    9. Save the table to data/preprocessed/spectrophoto
    '''
    try:
        for survey in ['SDSS', 'LAMOST']:
            # Open spectroscopy data
            req_cols = SDSS_LAMOST_SPECTRO_REQ_COLS[survey]
            if survey=='SDSS':
                df_spectro = pd.read_csv(SDSS_LAMOST_SPECTRO_FILEPATH[survey], delim_whitespace=True)[req_cols]
            else:
                with fits.open(SDSS_LAMOST_SPECTRO_FILEPATH[survey]) as hdul:
                    df_spectro = Table(hdul[1].data).to_pandas()[req_cols]
            logger.info(f'Original number of {survey.upper()} galaxies = {len(df_spectro)}')

            # Open the 2MASS data
            req_cols = ['dist_x', 'ra_01', 'dec_01', 'designation', 'glon', 'glat', 'j_ba', 'h_ba', 'k_ba', 
                        'sup_ba', 'r_ext', 'j_m_ext', 'h_m_ext', 'k_m_ext', 'j_r_eff', 'h_r_eff', 'k_r_eff']
            df_2mass = pd.read_csv(SDSS_LAMOST_TMASS_FILEPATH[survey], low_memory=False)[req_cols]

            # Merge FP + 2MASS and drop measurements without photometry (designation is null)
            logger.info(f"Merging {survey.upper()} spectroscopy with 2MASS photometry...")
            df = df_spectro.merge(df_2mass, left_index=True, right_index=True)
            df = df.dropna(subset='designation').rename({'designation': 'tmass', 'RA': 'ra', 'Dec': 'dec'}, axis=1)
            # Drop rows with duplicated 2MASS Id, pick the one with the smaller dist_x
            df = df.sort_values(by='dist_x', ascending=True)
            df = df.drop_duplicates(subset='tmass')
            # Rename 2MASS id column
            df['tmass'] = '2MASXJ' + df['tmass']
            logger.info(f"Remaining {survey.upper()} galaxies = {len(df)}")

            # Sanity test (check RAJ2000 DEJ2000 vs ra_01 dec_01)
            max_delta_ra = np.absolute(max(df['ra'] - df['ra_01']))
            max_delta_dec = np.absolute(max(df['dec'] - df['dec_01']))
            tol_ = 0.001
            logger.info(f'Max delta RA: {max_delta_ra}')
            logger.info(f'Max delta DEC: {max_delta_dec}')

            if (max_delta_ra > tol_) or (max_delta_dec > tol_):
                logger.info('The two tables do not match.')
                raise
            else:
                logger.info(f'The coordinates in {survey.upper()} and 2MASS response are consistent.')
                df = df.drop(['ra_01', 'dec_01'], axis=1)

            # Open John's measurements
            req_cols = ['tmass', 'log_r_h_app_j', 'log_r_h_smodel_j', 'log_r_h_model_j', 'fit_ok_j', 'red_chi_j',
                        'log_r_h_app_h', 'log_r_h_smodel_h', 'log_r_h_model_h', 'fit_ok_h',
                        'log_r_h_app_k', 'log_r_h_smodel_k', 'log_r_h_model_k', 'fit_ok_k']
            df_jrl = pd.read_csv(JRL_PHOTO_FILEPATH)[req_cols]

            # Merge SDSS_Spectro+2MASS and JRL photometry
            logger.info(f"Merging {survey.upper()}+2MASS with JRL photometry...")
            df = df.merge(df_jrl, on='tmass')
            logger.info(f'Remaining {survey.upper()} galaxies = {len(df)}')

            # Open cluster and group data
            ## Individual galaxies data
            req_cols = ['IDcl', 'RAJ2000', 'DEJ2000']
            with fits.open(TEMPEL_GAL_FILEPATH) as hdul:
                df_gal = Table(hdul[1].data).to_pandas()[req_cols]
            ## Group and cluster data
            req_cols = ['IDcl', 'zcl']
            with fits.open(TEMPEL_GROUP_FILEPATH) as hdul:
                df_gr = Table(hdul[1].data).to_pandas()[req_cols]
            ## Merge the two tables
            df_tempel = df_gal.merge(df_gr, on='IDcl', how='left')

            # Crossmatch SDSS/LAMOST data with Tempel data based on individual galaxy RA and DEC
            coords_mydata = SkyCoord(ra=df['ra'].to_numpy() * u.deg, dec=df['dec'].to_numpy() * u.deg)
            coords_tempel = SkyCoord(ra=df_tempel['RAJ2000'].to_numpy() * u.deg, dec=df_tempel['DEJ2000'].to_numpy() * u.deg)

            idx, sep2d, _ = coords_mydata.match_to_catalog_sky(coords_tempel)
            SEP_THRESH = 2.5
            is_counterpart = sep2d < SEP_THRESH * u.arcsec

            df['tempel_idx'] = idx
            df['tempel_counterpart'] = is_counterpart

            logger.info(f'Joining {survey.upper()} data with Tempel group and cluster data...')
            df = df.merge(df_tempel, left_on='tempel_idx', how='left', right_index=True).drop(['tempel_idx', 'RAJ2000', 'DEJ2000'], axis=1)
            logger.info(f'{survey.upper()} galaxies that are part of a cluster: {len(df[df.tempel_counterpart==True])}')

            # Save the resulting table
            logger.info(f'Number of galaxies = {len(df)}. Number of unique galaxies = {df.tmass.nunique()}. Saving the table to {SDSS_LAMOST_OUTPUT_FILEPATH[survey]}.')
            df.to_csv(SDSS_LAMOST_OUTPUT_FILEPATH[survey], index=False)

    except Exception as e:
        logger.error(f"Combining {survey.upper()} spectroscopy and photometry data failed. Reason: {e}")
    
def main():
    logger.info(f"{'='*50}")
    logger.info('Combining 6dFGS data...')
    start = time.time()
    combine_6df_spectrophoto()
    end = time.time()
    logger.info(f'Combining 6dFGS data successful! Time elapsed = {round(end-start, 2)} s')

    logger.info('\n')
    
    logger.info('Combining SDSS and LAMOST data...')
    start = time.time()
    combine_sdss_lamost_spectrophoto()
    end = time.time()
    logger.info(f'Combining SDSS and LAMOST data successful! Time elapsed = {round(end-start, 2)} s')

if __name__ == '__main__':
    main()
