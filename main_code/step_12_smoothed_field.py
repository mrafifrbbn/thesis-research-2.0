import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.utils.constants import *
from main_code.utils.CosmoFunc import *
from main_code.utils.logging_config import get_logger
from main_code.utils.filepaths import (
    LOGDIST_OUTPUT_FILEPATH
)

from astropy import units as u
from astropy.coordinates import SkyCoord

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create logging instance
logger = get_logger('fit_fp')

# Constants
SIGMA_FIDUCIAL = 4.0
SCALE = 3
SGZ_grids = [-20., 0., 60., 80.]
MAX_DIST = 161.2
METHOD = 'common_abc'


def combine_my_logdist():
    """Function to combine my logdist measurements and deduplicate 2MASS id by averaging
    """
    # Combine my data
    df = pd.DataFrame()
    for survey in SURVEY_LIST:
        df_temp = pd.read_csv(LOGDIST_OUTPUT_FILEPATH[survey])[
            ['tmass', 'ra', 'dec', 'z_dist_est', f'logdist_{METHOD}', f'logdist_err_{METHOD}']
        ].rename({
            f'logdist_{METHOD}': 'logdist',
            f'logdist_err_{METHOD}': 'logdist_err'
        }, axis=1)

        df = pd.concat([df, df_temp])

    logger.info(f"Number of rows: {len(df)} | Number of unique 2MASS ID: {df['tmass'].nunique()}")

    # Error-weighted average for repeat measurements
    df_dup = df.copy()
    df_dup = df_dup[df_dup.duplicated(subset='tmass', keep=False)]
    dup_tmass_id = df_dup['tmass'].unique().tolist()

    df_dup['w'] = df_dup['logdist_err'] ** (-2)
    df_dup['w_times_logdist'] = df_dup['w'] * df_dup['logdist']

    # Group by 2MASS ID and sum the weighted things
    df_grouped = df_dup.groupby(by='tmass', observed=False).agg(
        numerator=('w_times_logdist', 'sum'),
        denominator=('w', 'sum'),
        ra=('ra', 'mean'),
        dec=('dec', 'mean'),
        z_dist_est=('z_dist_est', 'mean'),
    )

    # Calculate error-weighted average of distance modulus and its error
    df_grouped['logdist_avg'] = df_grouped["numerator"] / df_grouped["denominator"]
    df_grouped['logdist_err_avg'] = 1 / np.sqrt(df_grouped["denominator"])
    df_grouped = df_grouped.reset_index()[['tmass', 'ra', 'dec', 'z_dist_est', 'logdist_avg', 'logdist_err_avg']].rename({
        'logdist_avg': 'logdist',
        'logdist_err_avg': 'logdist_err'
    }, axis=1)

    # Concatenate
    df = df[(~df['tmass'].isin(dup_tmass_id))]
    df = pd.concat([df, df_grouped])
    logger.info(f"Final number of rows: {len(df)}")

    return df


def get_supergalactic_coords(
        df: pd.DataFrame,
        ra_col_name: str = 'ra',
        dec_col_name: str = 'dec',
        z_dist_est_col_name: str = 'z_dist_est',
        H0: float = 100.0
    ) -> pd.DataFrame:
    
    # Fetch ra, dec, z
    ra = df[ra_col_name].to_numpy()
    dec = df[dec_col_name].to_numpy()
    z = df[z_dist_est_col_name].to_numpy()
    
    # Construct astropy coords object
    eq_coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    
    # Convert to supergalactic coordinates
    sgal_coords = eq_coords.supergalactic
    sgl = sgal_coords.sgl.radian
    sgb = sgal_coords.sgb.radian
    df['SGLON'] = np.rad2deg(sgl)
    df['SGLAT'] = np.rad2deg(sgb)
    
    # Get redshift-distance lookup table
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table(H0=H0)

    # The comoving distance to each galaxy using group redshift as distance indicator
    dz = sp.interpolate.splev(z, dist_spline, der=0)
    
    # Get supergalactic cartesian coordinates
    sgx = dz * np.cos(sgb) * np.cos(sgl)
    sgy = dz * np.cos(sgb) * np.sin(sgl)
    sgz = dz * np.sin(sgb)
    
    df['dz_dist_est'] = dz
    df['SGX'] = sgx
    df['SGY'] = sgy
    df['SGZ'] = sgz
    
    return df


def get_smoothing_length(coord_x, coord_y, coord_z, coords_data, sigma_fid=5.0, scale=5):
    """
    A function to calculate the smoothing length for every galaxy.
    """

    # Construct vector from the coordinates
    coord_j = np.array([coord_x, coord_y, coord_z])

    # Calculate distances from galaxy j to every other galaxy (including itself)
    distances = np.sqrt(np.sum((coord_j - coords_data)**2, axis=1))

    # Filter galaxies within scale * sigma_fid
    distances = distances[distances <= scale * sigma_fid]
    
    # Calculate local density for each galaxy
    delta_j = np.sum(np.exp(-0.5 * (distances / sigma_fid)**2))

    return delta_j


def generate_grids(
    x_span: float = 300.,
    y_span: float = 300.,
    z_list: list[float] = [-80., -30., 10., 70.], 
    grid_size: float = 4.0, 
    max_dist: float = 200.,
) -> np.ndarray:
    
    # Create grid coordinates
    grid_x = np.arange(-x_span, x_span + grid_size, grid_size)
    grid_y = np.arange(-y_span, y_span + grid_size, grid_size)
    grid_z = np.array(z_list)
    coords_grid = np.array(np.meshgrid(grid_x, grid_y, grid_z)).T.reshape(-1, 3)

    # Remove origin and edges
    grid_dist = np.sqrt(np.sum(coords_grid**2, axis=1))
    coords_grid = coords_grid[np.logical_and(grid_dist < max_dist, grid_dist > 0.)]

    # Convert to dataframe for easier processing
    df_grid = pd.DataFrame(coords_grid, columns=['SGX', 'SGY', 'SGZ'])
    x = df_grid['SGX'].to_numpy() * u.Mpc
    y = df_grid['SGY'].to_numpy() * u.Mpc
    z = df_grid['SGZ'].to_numpy() * u.Mpc

    # Create skycoord object from cartesian coordinates
    s = SkyCoord(sgx=x, sgy=y, sgz=z, frame='supergalactic', representation_type='cartesian')

    # 1. Remove Galactic plane (|b| > 10)
    g = s.transform_to('galactic')
    crit_gal_plane = np.absolute(g.b.value) > 10

    # 2. Remove area around the NCP (eyeballing/dirty masking)
    eq = s.transform_to('icrs')
    ra = eq.ra.value
    dec = eq.dec.value
    segment_1 = (ra <= 110) & (dec <= 50)
    segment_2 = ((ra > 110) & (ra <= 250)) & (dec <= 70)
    segment_3 = ((ra > 250) & (ra <= 305)) & (dec <= 55)
    segment_4 = ((ra > 305) & (ra <= 360)) & (dec <= 40)
    crit_ncp = segment_1 | segment_2 | segment_3 | segment_4

    # Gridpoints that will be considered
    df_grid_good = df_grid[crit_gal_plane & crit_ncp]

    # Return back as numpy array
    grid_good = df_grid_good.to_numpy()

    return grid_good


def main():

    logger.info(f"Smoothing velocity field with sigma_fiducial = {SIGMA_FIDUCIAL} and scale = {SCALE} at SGZ = {SGZ_grids} using method {METHOD}")
    try:
        # Get my (deduplicated) PV sample
        logger.info("Combining my logdists and deduplicating 2MASS IDs...")
        df = combine_my_logdist()

        # ---------------------------------- Calculate smoothing length for every galaxy ---------------------------------- #

        logger.info("Calculating smoothing length for every galaxy...")
        # Get supergalactic coordinates vector for each galaxy
        df = get_supergalactic_coords(df)
        coords_data = np.array([df['SGX'], df['SGY'], df['SGZ']]).T

        df['delta_j'] = df.apply(lambda x: get_smoothing_length(x.SGX, x.SGY, x.SGZ, coords_data, sigma_fid=SIGMA_FIDUCIAL, scale=SCALE), axis=1)
        df['sigma_j'] = 2 * SIGMA_FIDUCIAL * np.sqrt(np.exp(np.log(df['delta_j']).sum() / len(df)) / df['delta_j'])


        # ---------------------------------- Calculate smoothed logdist field ---------------------------------- #

        logger.info("Calculating smoothed velocity field for the grids")
        # Unpack data
        logdist = df['logdist'].to_numpy()
        logdist_err = df['logdist_err'].to_numpy()
        smoothing_length = df['sigma_j'].to_numpy()
        coords_data = np.array([df['SGX'], df['SGY'], df['SGZ']]).T

        # Create empty dataframe to store smoothed logdist field
        df_smoothed = pd.DataFrame()
        coords_grid = generate_grids(z_list=SGZ_grids, max_dist=MAX_DIST)
        df_smoothed[['SGX', 'SGY', 'SGZ']] = coords_grid

        # Determine N_j for each point i, calculate distance from every point to every galaxy
        distances = np.sqrt(np.sum((coords_grid[:, None] - coords_data)**2, axis=2))
        mask = distances <= SCALE * smoothing_length
        N_j = np.sum(mask, axis=1)

        # Calculate cosine of the angle between grid position and galaxy's position
        grid_dist = np.sqrt(np.sum(coords_grid**2, axis=1))
        gal_dist = np.sqrt(np.sum(coords_data**2, axis=1))
        cos_theta = np.sum(coords_grid[:, None] * coords_data, axis=2) / (grid_dist[:, None] * gal_dist)

        # Calculate the smoothed logdist for each grid and its estimated error
        weight = 1
        num_ = np.sum(mask * weight * logdist * cos_theta * np.exp(-0.5 * (distances / smoothing_length)**2) * (smoothing_length**(-3)), axis=1)
        num_err_ = np.sum(mask * weight * logdist_err * cos_theta * np.exp(-0.5 * (distances / smoothing_length)**2) * (smoothing_length**(-3)), axis=1)
        denom_ = np.sum(mask * weight * np.exp(-0.5 * (distances / smoothing_length)**2) * (smoothing_length**(-3)), axis=1)

        # Save to the dataframe
        df_smoothed['logdist_smoothed'] = num_ / denom_
        df_smoothed['logdist_smoothed_err'] = num_err_ / denom_

        # Remove NaNs
        df_smoothed = df_smoothed[(~df_smoothed['logdist_smoothed'].isna()) & (~df_smoothed['logdist_smoothed_err'].isna())]

        filepath = os.path.join(ROOT_PATH, f"artifacts/velocity_map/smoothed_s_{SCALE}_sfid_{SIGMA_FIDUCIAL}_{METHOD}.csv")
        df_smoothed.to_csv(filepath, index=False)

        logger.info("Smoothing velocity field successful!")
    except Exception as e:
        logger.error("Generating smoothed velocity map failed.", exc_info=True)


if __name__ == '__main__':
    main()
