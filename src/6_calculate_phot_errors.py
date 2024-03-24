import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.constants import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

# Create logging instance
logger = get_logger('zms_cut')

# Constant
MAG_BIN_WIDTH = 0.2    # magnitude bin width

GLOBAL_REQ_COL = ['ra', 'dec', 'zhelio', 'z_cmb', 'z_dist_est', 'j_m_ext', 'extinction_j', 'kcor_j', 'r_j', 'er_j', 's_scaled', 'es_scaled', 'i_j', 'ei_j']
REQ_COLS = {
    '6dFGS': ['tmass', '_6dFGS'] + GLOBAL_REQ_COL,
    'SDSS': ['tmass', 'objid'] + GLOBAL_REQ_COL,
    'LAMOST': ['tmass', 'obsid'] + GLOBAL_REQ_COL
}

INPUT_FILEPATH = {
    '6dFGS': 'data/processed/zms_cut/6dfgs.csv',
    'SDSS': 'data/processed/zms_cut/sdss.csv',
    'LAMOST': 'data/processed/zms_cut/lamost.csv'
}

OUTPUT_FILEPATH = {
    '6dFGS': 'data/foundation/fp_sample/6dfgs.csv',
    'SDSS': 'data/foundation/fp_sample/sdss.csv',
    'LAMOST': 'data/foundation/fp_sample/lamost.csv'
}

# Piecewise linear function
def piecewise_linear(x, x0, y0, k):
    '''
    A function that is constant at x <= x0, and linear at x > x0.
    '''
    y_pred = np.piecewise(x, [x <= x0, x > x0], [lambda x: y0, lambda x: k * (x - x0) + y0])
    return y_pred

# Function to derive the photometric error
def derive_phot_error():
    '''
    A function to derive the photometric errors following the method outlined in Magoulas et al. 2012.
    The scatter vs magnitude is fit with the piecewise linear function iteratively to remove outliers.
    '''
    df = pd.DataFrame()
    for survey in SURVEY_LIST:
        req_cols = ['tmass', 'j_m_ext', 'extinction_j', 'i_j', 'i_h', 'i_k']
        df_survey = pd.read_csv(INPUT_FILEPATH[survey])[req_cols]
        df = pd.concat([df, df_survey])

    df = df.drop_duplicates(subset='tmass')

    # Calculate extinction-corrected J-band magnitude
    df['mag_j'] = df['j_m_ext'] - df['extinction_j']

    # Calculate all colors
    df['color_jh'] = df['i_j'] - df['i_h']
    df['color_jk'] = df['i_j'] - df['i_k']
    df['color_hk'] = df['i_h'] - df['i_k']

    # Create the magnitude bins
    bin_list = np.arange(df['mag_j'].min(), df['mag_j'].max() + MAG_BIN_WIDTH, MAG_BIN_WIDTH)
    bin_label = range(1, len(bin_list))
    df['mag_j_bin'] = pd.cut(df['mag_j'], bin_list, labels=bin_label)

    # Calculate the variance of the colors in each bin
    delta2_jh = df.groupby('mag_j_bin', observed=False)['color_jh'].var()
    delta2_jk = df.groupby('mag_j_bin', observed=False)['color_jk'].var()
    delta2_hk = df.groupby('mag_j_bin', observed=False)['color_hk'].var()
    mag_j_bin_mean = df.groupby('mag_j_bin', observed=False)['mag_j'].mean()

    # Calculate the error in j alone
    e_i_j = np.sqrt(0.5 * (delta2_jh + delta2_jk - delta2_hk))

    # Remove nan values
    notnan_indices = ~np.isnan(e_i_j)
    e_i_j = e_i_j[notnan_indices].to_numpy()
    mag_j_bin_mean = mag_j_bin_mean[notnan_indices].to_numpy()

    # Fit the function iteratively
    x_data = mag_j_bin_mean
    y_data = e_i_j
    datacount = len(y_data)
    is_converged = False
    i = 1
    while not is_converged:
        logger.info(f"Iteration {i}")
        # Fit the parameters
        popt, pcov = curve_fit(piecewise_linear, x_data, y_data, p0=[11.0, 0.02, 0.05])
        logger.info(f'Parameters: x0 = {round(popt[0], 4)}')
        logger.info(f'y0 = {round(popt[1], 4)}')
        logger.info(f'k = {round(popt[2], 4)}')
        logger.info(f'Constant = {round(popt[1] - popt[2] * popt[0], 4)}')

        # Calculate the predicted values and chi statistics
        y_pred = piecewise_linear(x_data, *popt)
        chisq = ((y_data - y_pred) / y_pred)**2

        # Reject the 'bad' data (chisq > 0.5)
        bad_data_indices = chisq > 0.06
        x_data = x_data[~bad_data_indices]
        y_data = y_data[~bad_data_indices]
        datacount_new = len(y_data)

        is_converged = True if datacount == datacount_new else False
        datacount = datacount_new
        i += 1
    
    # Print the final equation
    logger.info(f"{'-' * 10} Final results {'-' * 10}")
    constant_pl = round(popt[1] - popt[2] * popt[0], 4)
    logger.info(f"Equation: y = {popt[1]:.4f} for x <= {popt[0]:.4f}")
    logger.info(f"Equation: y = {popt[2]:.4f}m_J + ({constant_pl:.4f}) for x > {popt[0]:.4f}")

    # Plot the results
    x_trial = np.linspace(np.min(x_data) - 0.5, np.max(x_data) + 0.1, 1000)
    y_trial = piecewise_linear(x_trial, *popt)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.scatter(x_data, y_data)
    ax.plot(x_trial, y_trial, color='k')
    ax.set_xlabel(r'$m_J$ (mag)', fontsize=14)
    ax.set_ylabel(r'$e_{i,J}$ [$L_\odot\ \text{pc}^{-2}$]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(ls=':', alpha=0.5)
    plt.tight_layout()
    img_output_path = os.path.join(ROOT_PATH, f'img/phot_error.png')
    fig.savefig(img_output_path, dpi=300)
    
    return popt
    
def apply_phot_error(popt):
    '''
    This function applies the error formula for every galaxy.
    '''
    for survey in SURVEY_LIST:
        df = pd.read_csv(INPUT_FILEPATH[survey])
        df['ei_j'] = piecewise_linear(np.array(df['j_m_ext'] - df['extinction_j']), *popt)
        df['er_j'] = 0.5 * df['ei_j']
        df = df[REQ_COLS[survey]].rename({'r_j': 'r', 'er_j': 'er', 's_scaled': 's', 'es_scaled': 'es', 'i_j': 'i', 'ei_j': 'ei'}, axis=1)
        df.to_csv(OUTPUT_FILEPATH[survey], index=False)
        
def main():
    try:
        logger.info(f"{'=' * 100}")
        logger.info("Deriving photometric errors...")
        popt = derive_phot_error()
        logger.info("Deriving photometric errors successful!")
        
        logger.info("Applying photometric errors and selecting the final columns...")
        apply_phot_error(popt)
        logger.info("Applying photometric errors and selecting the final columns successful!")
        logger.info('\n')
    except Exception as e:
        logger.error(f'Deriving photometric errors failed. Reason: {e}.')
    
if __name__ == '__main__':
    main()