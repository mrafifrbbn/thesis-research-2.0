import os
import numpy as np
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Plot configurations
GOLDEN_RATIO = 1.618033988749895
FIGURE_HEIGHT = 5
FIGURE_WIDTH = FIGURE_HEIGHT * GOLDEN_RATIO
DEFAULT_FIGSIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

# Survey names
SURVEY_LIST = ['6dFGS', 'SDSS', 'LAMOST']

# Speed of light (km/s)
LIGHTSPEED = 299792.458
# Matter density parameter
OMEGA_M = 0.3121

# Sample selection constants
MAG_LOW = 8.0
MAG_HIGH = 13.65
ZMIN = 3000.0 / LIGHTSPEED
ZMAX = 16120. / LIGHTSPEED

# Galactic extinction constants in the JHK bands (Schlafly and Finkbeiner)
EXTINCTION_CONSTANT = {
    'j': 0.723,
    'h': 0.460,
    'k': 0.310
}

# Sun absolute magnitude constants (Wilmer?)
SOLAR_MAGNITUDE = {
    'j': 3.67,
    'h': 3.32,
    'k': 3.27
}

# # Nominal veldisp limit for each survey [km/s]
# SURVEY_VELDISP_LIMIT = {
#     '6dFGS': np.log10(112),
#     'SDSS': np.log10(70),
#     'LAMOST': np.log10(50)
# }

# Convert integers to string (1 -> first, 2 -> second, etc.)
INT_CONVERTER = {
    0: 'default',
    1: 'first',
    2: 'second',
    3: 'third'
}

# Grab veldisp offsets
# try:
#     totoff = pd.read_csv(os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/totoffs.csv'))
#     off_6df = totoff.loc[0, ['off_6df']].values[0]
#     off_sdss = totoff.loc[0, ['off_sdss']].values[0]
#     off_lamost = totoff.loc[0, ['off_lamost']].values[0]
# except:
#     off_6df = 0
#     off_sdss = 0
#     off_lamost = 0

totoff = pd.read_csv(os.path.join(ROOT_PATH, 'artifacts/veldisp_calibration/totoffs.csv'))
off_6df = totoff.loc[0, ['off_6df']].values[0]
off_sdss = totoff.loc[0, ['off_sdss']].values[0]
off_lamost = totoff.loc[0, ['off_lamost']].values[0]

# Define the veldisp lower limit (as defined in the guide)
SURVEY_VELDISP_LIMIT = {
    # Default: use nominal veldisp limit + offset of each survey
    0: {
        '6dFGS': np.log10(112) - off_6df,
        'SDSS': np.log10(70) - off_sdss,
        'LAMOST': np.log10(50) - off_lamost
    },
    # First setting: use 6dFGS veldisp + offset for everything
    1: {
        '6dFGS': np.log10(112) - off_6df,
        'SDSS': np.log10(112) - off_6df,
        'LAMOST': np.log10(112) - off_6df
    },
    # Second setting: use 6dFGS veldisp + offset for 6dFGS and SDSS and LAMOST veldisp + offset for LAMOST
    2: {
        '6dFGS': np.log10(112) - off_6df,
        'SDSS': np.log10(112) - off_6df,
        'LAMOST': np.log10(50) - off_lamost
    }
}

# Define the FP setting to derive galaxy's logdists
SURVEY_FP_SETTING = {
    # Default: use each survey's own FP fit
    0: {
        '6dFGS': '6dFGS',
        'SDSS': 'SDSS',
        'LAMOST': 'LAMOST',
        'ALL_COMBINED': 'ALL_COMBINED'
    },
    # First setting: use 6dFGS FP for all
    1: {
        '6dFGS': '6dFGS',
        'SDSS': '6dFGS',
        'LAMOST': '6dFGS',
        'ALL_COMBINED': '6dFGS'
    },
    # Second setting: use SDSS FP for all
    2: {
        '6dFGS': 'SDSS',
        'SDSS': 'SDSS',
        'LAMOST': 'SDSS',
        'ALL_COMBINED': 'SDSS'
    }
}

# Function to create parent folder, given a full absolute path as dictionary or string
def create_parent_folder(full_abspath) -> None:
    if isinstance(full_abspath, dict):
        for filepath in full_abspath.values():
            output_filepath = Path(filepath)
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
    elif isinstance(full_abspath, str):
        output_filepath = Path(full_abspath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)