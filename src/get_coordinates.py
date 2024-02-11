import os
import pandas as pd
from astropy.io import fits
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')
SDFGS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/campbell_table8.ascii')
SDSS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/sdss/SDSS_spectro_mrafifrbbn.csv')
LAMOST_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')

def get_eq_coords():
    '''
    Combining the equatorial coordinates for all the galaxies in the raw data.
    These coordinates are fed into the 2MASS XSC to obtain the 2MASS photometry.
    '''

    # Create logging instance
    logger = get_logger('get_eq_coords')
    logger.info('Fetching sky coordinates from the raw data sources...')

    # 6dFGS galaxies: ra still in hour, so need to convert to degrees
    df1 = pd.read_csv(SDFGS_FILEPATH, delim_whitespace=True)[['ra', 'dec']]
    df1['ra'] = df1['ra']*15
    output_path_6dfgs = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/6dfgs.csv')
    logger.info(f'Total 6dFGS galaxies: {len(df1)}. Saving the coordinates to {output_path_6dfgs}')
    df1.to_csv(output_path_6dfgs, index=False)

    # SDSS galaxies
    df2 = pd.read_csv(SDSS_FILEPATH)[['ra', 'dec']]
    output_path_sdss = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/sdss.csv')
    logger.info(f'Total SDSS galaxies: {len(df2)}. Saving the coordinates to {output_path_sdss}')
    df2.to_csv(output_path_sdss, index=False)

    # LAMOST galaxies
    with fits.open(LAMOST_FILEPATH) as hdul:
        df3 = pd.DataFrame(hdul[1].data)[['ra', 'dec']]
    output_path_lamost = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/lamost.csv')
    logger.info(f'Total LAMOST galaxies: {len(df3)}. Saving the coordinates to {output_path_lamost}')
    df3.to_csv(output_path_lamost, index=False)

    return

if __name__ == '__main__':
    get_eq_coords()

