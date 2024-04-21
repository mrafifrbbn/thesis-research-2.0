import numpy as np
from pathlib import Path

# Plot configurations
GOLDEN_RATIO = 1.618033988749895
FIGURE_HEIGHT = 5
FIGURE_WIDTH = FIGURE_HEIGHT * GOLDEN_RATIO
DEFAULT_FIGSIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

# Survey names
SURVEY_LIST = ['6dFGS', 'SDSS', 'LAMOST']

# Speed of light (km/s)
LIGHTSPEED = 299792.458

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

# Nominal veldisp limit for each survey [km/s]
SURVEY_VELDISP_LIMIT = {
    '6dFGS': np.log10(112),
    'SDSS': np.log10(70),
    'LAMOST': np.log10(50)
}

# Convert integers to string (1 -> first, 2 -> second, etc.)
INT_CONVERTER = {
    0: 'default',
    1: 'first',
    2: 'second',
    3: 'third'
}

# Function to create parent folder, given a full absolute path as dictionary or string
def create_parent_folder(full_abspath):
    if isinstance(full_abspath, dict):
        for filepath in full_abspath.values():
            output_filepath = Path(filepath)
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
    elif isinstance(full_abspath, str):
        output_filepath = Path(full_abspath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)