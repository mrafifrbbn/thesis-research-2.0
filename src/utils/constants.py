# Plot configurations
GOLDEN_RATIO = 1.618033988749895
FIGURE_HEIGHT = 5
FIGURE_WIDTH = FIGURE_HEIGHT*GOLDEN_RATIO
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