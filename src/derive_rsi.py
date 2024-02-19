import os
import time
import numpy as np
import pandas as pd
from scipy import interpolate

from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from dustmaps.config import config
config['data_dir'] = 'etc/dustmaps'

from utils.constants import *
from utils.logging_config import get_logger
from utils.helio_cmb import perform_corr
from utils.calc_kcor import calc_kcor
from utils.CosmoFunc import rz_table

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('get_spectrophoto')

# Primary key column names for SDSS and LAMOST
SURVEY_ID_COL_NAME = {
    'SDSS': 'objID',
    'LAMOST': 'obsid'
}

# Aperture size and veldisp column names
SPECTROSCOPY_CONFIG = {
    '6dFGS': {
        'aperture_size': 3.35,
        'veldisp_col_name': 'Vd',
        'veldisp_err_col_name': 'e_Vd'
    },
    'SDSS': {
        'aperture_size': 1.5,
        'veldisp_col_name': 'sigmaStars',
        'veldisp_err_col_name': 'sigmaStarsErr'
    },
    'LAMOST': {
        'aperture_size': 1.65,
        'veldisp_col_name': 'veldisp',
        'veldisp_err_col_name': 'veldisp_err'
    }
}

# File paths
SPECTROPHOTO_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/preprocessed/spectrophoto/lamost.csv')
}

OUTPUT_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/processed/rsi_derived/lamost.csv')
}

def derive_rsi():
    '''
    A function to derive r, s, i quantities for each galaxy in the three surveys.
    '''

    try:
        for survey in SURVEY_LIST:

            # Open spectroscopy+photometry file
            df = pd.read_csv(SPECTROPHOTO_FILEPATHS[survey])
            logger.info(f'Original number of {survey} galaxies = {len(df)} galaxies.')

            # Step 0: extra step only for SDSS and LAMOST (John's selection criteria)
            if survey in ['SDSS', 'LAMOST']:
                
                # Add prefixes to the survey ID's to convert them to object type
                # Only for SDSS and LAMOST as 6dFGS ID's are already object type
                df[SURVEY_ID_COL_NAME[survey]] = survey + df[SURVEY_ID_COL_NAME[survey]].astype(str)
                
                # Select only good J-band radii
                # Selection criteria (from John): fit_OK, r_model > 1 arcsec, reduced chi <= 2.
                df = df[df['fit_ok_j'] == 'OK']
                logger.info(f"Selected galaxies with fit_ok_j == 'OK'. Remaining galaxies = {len(df)}")
                
                df = df[df['log_r_h_model_j'] > 0.]
                logger.info(f"Selected galaxies with log_r_h_model_j > 0. Remaining galaxies = {len(df)}")
                
                df = df[df['red_chi_j'] <= 2.]
                logger.info(f"Selected galaxies with red_chi_j <= 2. Remaining galaxies = {len(df)}")

            # Step 1: derive PSF-corrected radii
            logger.info('Deriving PSF-corrected radii...')
            for band in 'jhk':
                if survey == '6dFGS':
                    df[f'theta_{band}'] = 10 ** df[f'{band.upper()}logr']
                else:
                    df[f'delta_r_{band}'] = 10 ** (df[f'log_r_h_smodel_{band}']) - 10 ** (df[f'log_r_h_model_{band}'])
                    df[f'theta_{band}'] = 10 ** (df[f'log_r_h_app_{band}']) - df[f'delta_r_{band}']

            # Step 2: calculate CMB frame redshift for individual galaxies (also rederive for 6dFGS)
            logger.info('Calculating CMB frame redshift for each galaxy...')
            df['z_cmb'] = perform_corr(df['z'], df['ra'], df['dec'], corrtype='full', dipole='Planck')
            
            # Step 3: use group/cluster redshift for galaxies in group/cluster
            logger.info('Obtaining group/cluster mean redshift if available...')
            if survey in ['SDSS', 'LAMOST']:
                df['z_dist_est'] = np.where(df['tempel_counterpart'] == True, df['zcl'], df['z_cmb'])
            else:
                df['z_dist_est'] = np.where(df['cz_gr'] != 0., df['cz_gr'] / LIGHTSPEED, df['z_cmb'])

            # Step 4: aperture size corrections for the velocity dispersions
            logger.info('Calculating aperture size-corrected velocity dispersions...')
            # Convert J-band radii to R-band radii
        #     R_j = 10 ** (1.029 * np.log10(df['theta_j']) + 0.140)
            R_j = 10 ** (1.029 * np.log10(df['theta_j'] * np.sqrt(df['j_ba'])) + 0.140)
            
            aperture_size = SPECTROSCOPY_CONFIG[survey]['aperture_size']
            veldisp_col = SPECTROSCOPY_CONFIG[survey]['veldisp_col_name']
            veldisp_err_col = SPECTROSCOPY_CONFIG[survey]['veldisp_err_col_name']
            
            df['sigma_corr'] = df[veldisp_col] * ((R_j / 8) / aperture_size) ** (-0.04)
            df['e_sigma_corr'] = df[veldisp_err_col] * ((R_j / 8) / aperture_size) ** (-0.04)
            
            # Step 5: calculate Galactic extinctions in the JHK bands
            logger.info('Calculating Galactic extinctions in the JHK band...')
            sfd = SFDQuery()
            coords = SkyCoord(df['ra'], df['dec'], unit='deg', frame ='fk5')
            ebv = sfd(coords)
            for band in 'jhk':
                extinction_constant = EXTINCTION_CONSTANT[band]
                df[f'extinction_{band}'] = extinction_constant * ebv

            # Step 6: calculate k-corrections
            logger.info('Calculating K-corrections...')
            z = df['z'].to_numpy()
            color_J2H2 = (df['j_m_ext'] - df['extinction_j']) - (df['h_m_ext'] - df['extinction_h']).to_numpy()
            color_J2Ks2 = (df['j_m_ext'] - df['extinction_j']) - (df['k_m_ext'] - df['extinction_k']).to_numpy()
            df['kcor_j'] = calc_kcor('J2', z, 'J2 - H2', color_J2H2)
            df['kcor_h'] = calc_kcor('H2', z, 'J2 - H2', color_J2H2)
            df['kcor_k'] = calc_kcor('Ks2', z, 'J2 - Ks2', color_J2Ks2)

            # Step 7: derive r and i
            logger.info('Deriving r and i...')
            ## Get redshift-distance lookup table
            red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table()
            ## Comoving distance for individual galaxies
            dz = interpolate.splev(df["z_cmb"].to_numpy(), dist_spline)
            ## Comoving distance for group galaxies
            dz_cluster = interpolate.splev(df["z_dist_est"].to_numpy(), dist_spline)
            ## Use the circularized effective radii
            for band in 'jhk':
                circularized_radius = df[f'theta_{band}'] * np.sqrt(df[f'{band}_ba'])
                df[f'r_{band}'] = np.log10(circularized_radius) + np.log10(dz_cluster) \
                                + np.log10(1000.0 * np.pi / (180.0 * 3600.0)) - np.log10(1.0 + df['z'].to_numpy())
                
                df[f'i_{band}'] = 0.4 * SOLAR_MAGNITUDE[band] - 0.4 * df[f'{band}_m_ext'] - np.log10(2.0 * np.pi) \
                                - 2.0 * np.log10(circularized_radius) + 4.0 * np.log10(1.0 + df['z']) \
                                + 0.4 * df[f'kcor_{band}'] + 0.4 * df[f'extinction_{band}'] \
                                + 2.0 * np.log10(180.0 * 3600.0 / (10.0*np.pi))

            # Step 7: derive s
            logger.info('Deriving s...')
            df['s'] = np.log10(df['sigma_corr'])
            df['es'] = (df['e_sigma_corr'] / df['sigma_corr']) / np.log(10)
            
            # Step 8: save the data
            logger.info(f'Saving the output at {OUTPUT_FILEPATHS[survey]}...')
            df.to_csv(OUTPUT_FILEPATHS[survey],index=False)
    
    except Exception is e:
        logger.error(f'Deriving r, s, i for {survey} failed. Reason: {e}')

if __name__ == '__main__':
    derive_rsi()