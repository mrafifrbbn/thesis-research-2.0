import os
import time
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

# Create logging instance
logger = get_logger('get_eq_coords')

# File paths
ROOT_PATH = os.environ.get('ROOT_PATH')
SDFGS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_fp_vizier.fits')
SDSS_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/sdss/SDSS_spectro.csv')
LAMOST_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')

def get_eq_coords():
    '''
    Combining the equatorial coordinates for all the galaxies in the raw data and save them in IPAC format.
    These coordinates are fed into the 2MASS XSC to obtain the 2MASS photometry.
    '''
    try:
        # 6dFGS galaxies: ra still in hour, so need to convert to degrees
        with fits.open(SDFGS_FILEPATH) as hdul:
            df1 = Table(hdul[1].data).to_pandas()[['RAJ2000', 'DEJ2000']]
        df1['RAJ2000'] *= 15
        df1 = df1.rename({'RAJ2000': 'ra', 'DEJ2000': 'dec'}, axis=1)
        output_path = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/6dfgs.ascii')
        logger.info(f'Total 6dFGS galaxies: {len(df1)}. Saving the coordinates to {output_path}')
        Table.from_pandas(df1).write(output_path, format='ipac', overwrite=True)

        # SDSS galaxies
        df2 = pd.read_csv(SDSS_FILEPATH)[['ra', 'dec']]
        output_path = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/sdss.ascii')
        logger.info(f'Total SDSS galaxies: {len(df2)}. Saving the coordinates to {output_path}')
        Table.from_pandas(df2).write(output_path, format='ipac', overwrite=True)

        # LAMOST galaxies
        with fits.open(LAMOST_FILEPATH) as hdul:
            df3 = pd.DataFrame(hdul[1].data)[['ra', 'dec']]
        output_path = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/lamost.ascii')
        logger.info(f'Total LAMOST galaxies: {len(df3)}. Saving the coordinates to {output_path}')
        Table.from_pandas(df3).write(output_path, format='ipac', overwrite=True)
    
    except Exception as e:
        logger.error(f'Fetching coordinates failed. Reason: {e}')

def main():

    logger.info('Fetching sky coordinates from the raw data sources...')
    start = time.time()
    get_eq_coords()
    end = time.time()
    logger.info(f'Fetching sky coordinates from the raw data sources successful! Time elapsed: {round(end - start, 2)} s.')

if __name__ == '__main__':
    main()

