import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Environment variable
ROOT_PATH = os.environ.get('ROOT_PATH')
SMIN_SETTING = os.environ.get('SMIN_SETTING')
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Filepaths for raw spectroscopy data
SPECTROSCOPY_RAW_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/raw/6dfgs/sdfgs_fp_vizier.fits'),
    'SDSS': os.path.join(ROOT_PATH, 'data/raw/sdss/sdss_howlett2022.dat'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/raw/lamost/lamost_DR7_VDcat_20200825.fits')
}

# Filepaths for sky coordinates (RA, DEC)
SKY_COORDS_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/6dfgs.ascii'),
    'SDSS': os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/sdss.ascii'),
    'LAMOST': os.path.join(ROOT_PATH, 'data/preprocessed/sky_coord/lamost.ascii')
}

# Foundation zone FP sample filepaths
FOUNDATION_ZONE_FP_SAMPLE_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss_lamost.csv'),
    '6dFGS_SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs_sdss.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/all_combined.csv')
}

# Outlier-rejected FP sample filepaths
OUTLIER_REJECT_FP_SAMPLE_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/lamost.csv'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss_lamost.csv'),
    '6dFGS_SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs_sdss.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/all_combined.csv')
}

# FP fits filepath
FP_FIT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/fp_fits.csv')

# FP fits MCMC chain filepaths
MCMC_CHAIN_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs_chain.npy'),
    'SDSS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss_chain.npy'),
    'LAMOST': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/lamost_chain.npy'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss_lamost_chain.npy'),
    '6dFGS_SDSS': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs_sdss_chain.npy'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/all_combined_chain.npy')
}

# FP fits likelihood corner plot filepath
FP_FIT_LIKELIHOOD_CORNERPLOT_FILEPATH = os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/three_surveys_likelihood.png')

# FP fits likelihood distributions filepaths
FP_FIT_LIKELIHOOD_DISTRIBUTION_FILEPATHS = {
    '6dFGS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs.png'),
    'SDSS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss.png'),
    'LAMOST': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/lamost.png'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/sdss_lamost.png'),
    '6dFGS_SDSS': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/6dfgs_sdss.png'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'img/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/all_combined.png')
}

# FP fits typical scatter calculation results filepath
FP_FIT_TYPICAL_SCATTER_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{FP_FIT_METHOD}/fp_scatter.csv')