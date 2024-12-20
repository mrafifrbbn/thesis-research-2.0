import os
import numpy as np
import pandas as pd

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
NEW_SURVEY_LIST = SURVEY_LIST + ['SDSS_LAMOST', 'ALL_COMBINED']

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

# Completeness plot
DEFAULT_MARKERSTYLES = {
    "6dFGS": "v",
    "SDSS": "s",
    "LAMOST": "D"
}
COMPLETENESS_BIN_WIDTH = 0.15

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
        'LAMOST': np.log10(112) - off_6df,
        '6dFGS_SDSS': np.log10(112) - off_6df,
        'SDSS_LAMOST': np.log10(112) - off_6df,
        'ALL_COMBINED': np.log10(112) - off_6df
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
    },
    # Third setting: use 6dFGS for 6dFGS and SDSS for SDSS and LAMOST
    3: {
        '6dFGS': '6dFGS',
        'SDSS': 'SDSS',
        'LAMOST': 'SDSS',
        'ALL_COMBINED': 'ALL_COMBINED'
    }
}

# Constants when fitting the FP
PVALS_CUT = 0.001
REJECT_OUTLIERS = True
PARAM_BOUNDARIES = [(1.2, 1.6), (-1.1, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.0, 0.06), (0.25, 0.45), (0.14, 0.25)]

r_label = r'$\log_{10}R_e$'
r_unit = r'$h^{-1}$ kpc'
s_label = r'$\log_{10}\sigma_0$'
s_unit = r'$\mathrm{km}\ \mathrm{s}^{-1}$'
i_label = r'$\log_{10}I_e$'
i_unit = r'$\mathrm{L}_\odot\ \mathrm{pc}^{-2}$'