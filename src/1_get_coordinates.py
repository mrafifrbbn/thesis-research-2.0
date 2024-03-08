import os
import time
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from utils.logging_config import get_logger
from dotenv import load_dotenv
load_dotenv()

# Environment variable
ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('get_eq_coords')

# Input file paths
SDFGS_INPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_fp_vizier.fits')
SDSS_INPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/sdss/sdss_howlett2022.dat')
LAMOST_INPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')

# Output file paths
SDFGS_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/6dfgs.ascii')
SDSS_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/sdss.ascii')
LAMOST_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/lamost.ascii')

def get_eq_coords():
    '''
    Combining the equatorial coordinates for all the galaxies in the raw data and save them in IPAC format.
    These coordinates are fed into the 2MASS XSC to obtain the 2MASS photometry.
    '''
    # 6dFGS galaxies: ra still in hour, so need to convert to degrees
    with fits.open(SDFGS_INPUT_FILEPATH) as hdul:
        df1 = Table(hdul[1].data).to_pandas()[['RAJ2000', 'DEJ2000']]
    df1['RAJ2000'] *= 15
    df1 = df1.rename({'RAJ2000': 'ra', 'DEJ2000': 'dec'}, axis=1)
    logger.info(f'Total 6dFGS galaxies: {len(df1)}. Saving the coordinates to {SDFGS_OUTPUT_FILEPATH}')
    Table.from_pandas(df1).write(SDFGS_OUTPUT_FILEPATH, format='ipac', overwrite=True)

    # SDSS galaxies
    df2 = pd.read_csv(SDSS_INPUT_FILEPATH, delim_whitespace=True)[['RA', 'Dec']]
    df2 = df2.rename({'RA': 'ra', 'Dec': 'dec'}, axis=1)
    logger.info(f'Total SDSS galaxies: {len(df2)}. Saving the coordinates to {SDSS_OUTPUT_FILEPATH}')
    Table.from_pandas(df2).write(SDSS_OUTPUT_FILEPATH, format='ipac', overwrite=True)

    # LAMOST galaxies
    with fits.open(LAMOST_INPUT_FILEPATH) as hdul:
        df3 = pd.DataFrame(hdul[1].data)[['ra', 'dec']]
    logger.info(f'Total LAMOST galaxies: {len(df3)}. Saving the coordinates to {LAMOST_OUTPUT_FILEPATH}')
    Table.from_pandas(df3).write(LAMOST_OUTPUT_FILEPATH, format='ipac', overwrite=True)

def main():
    try:
        logger.info('Fetching sky coordinates from the raw data sources...')
        start = time.time()
        get_eq_coords()
        end = time.time()
        logger.info(f'Fetching sky coordinates from the raw data sources successful! Time elapsed: {round(end - start, 2)} s.')
    except Exception as e:
        logger.error(f'Fetching coordinates failed. Reason: {e}')

if __name__ == '__main__':
    main()

