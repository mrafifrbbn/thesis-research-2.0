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
OUTPUT_PATH = os.path.join(ROOT_PATH, 'data/raw/sky_coord/sky_coord.csv')

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
    logger.info(f'Total 6dFGS coordinates: {len(df1)}')

    # SDSS galaxies
    df2 = pd.read_csv(SDSS_FILEPATH)[['ra', 'dec']]
    logger.info(f'Total SDSS coordinates: {len(df2)}')

    # LAMOST galaxies
    with fits.open(LAMOST_FILEPATH) as hdul:
        df3 = pd.DataFrame(hdul[1].data)[['ra', 'dec']]
    logger.info(f'Total LAMOST coordinates: {len(df3)}')
        
    # Concatenate the three coordinates
    df = pd.concat([df1, df2, df3])
    logger.info(f'Total coordinate obtained: {len(df)}')

    # Save the coordinates as csv file
    logger.info(f'Saving the coordinates at {OUTPUT_PATH}')
    df.to_csv(OUTPUT_PATH, index=False)

    return

if __name__ == '__main__':
    get_eq_coords()

