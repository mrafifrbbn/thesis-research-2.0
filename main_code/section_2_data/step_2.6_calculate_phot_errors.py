import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict

from utils.constants import *
from utils.logging_config import get_logger

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))

# Create logging instance
logger = get_logger('zms_cut')

from main_code.utils.functions import create_parent_folder

# Constant
MAG_BIN_WIDTH = 0.2    # magnitude bin width

GLOBAL_REQ_COL = ['ra', 'dec', 'zhelio', 'z_cmb', 'z_dist_est', 'j_m_ext', 'extinction_j', 'kcor_j', 'r_j', 'er_j', 's_scaled', 'es_scaled', 'i_j', 'ei_j', 'Group', 'Nr']
REQ_COLS = {
    '6dFGS': ['tmass', '_6dFGS'] + GLOBAL_REQ_COL,
    'SDSS': ['tmass', 'objid'] + GLOBAL_REQ_COL,
    'LAMOST': ['tmass', 'obsid'] + GLOBAL_REQ_COL
}

INPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/processed/zms_cut/smin_setting_{SMIN_SETTING}/lamost.csv')
}

OUTPUT_FILEPATH = {
    '6dFGS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs.csv'),
    'SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss.csv'),
    'LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/lamost.csv'),
    'SDSS_LAMOST': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/sdss_lamost.csv'),
    '6dFGS_SDSS': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/6dfgs_sdss.csv'),
    'ALL_COMBINED': os.path.join(ROOT_PATH, f'data/foundation/fp_sample/smin_setting_{SMIN_SETTING}/all_combined.csv')
}
create_parent_folder(OUTPUT_FILEPATH)

MODEL_ARTIFACT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/phot_error/smin_setting_{SMIN_SETTING}/model.csv')
create_parent_folder(MODEL_ARTIFACT_FILEPATH)

XDATA_ARTIFACT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/phot_error/smin_setting_{SMIN_SETTING}/xdata.npy')
YDATA_ARTIFACT_FILEPATH = os.path.join(ROOT_PATH, f'artifacts/phot_error/smin_setting_{SMIN_SETTING}/ydata.npy')
create_parent_folder(XDATA_ARTIFACT_FILEPATH)
create_parent_folder(YDATA_ARTIFACT_FILEPATH)

IMG_OUTPUT_FILEPATH = os.path.join(ROOT_PATH, f'img/phot_error/smin_setting_{SMIN_SETTING}.png')

COMPLETENESS_ARTIFACT_PATH = os.path.join(ROOT_PATH, f"artifacts/magnitude_completeness/smin_setting_{SMIN_SETTING}/completeness_model.csv")
create_parent_folder(COMPLETENESS_ARTIFACT_PATH)

COMPLETENESS_IMAGE_PATH = os.path.join(ROOT_PATH, f"img/magnitude_completeness/smin_setting_{SMIN_SETTING}/completeness_model.png")
create_parent_folder(COMPLETENESS_IMAGE_PATH)

# Piecewise linear function
def piecewise_linear(x: np.ndarray, x0: float, y0: float, k: float) -> np.ndarray:
    '''
    A function that is constant at x <= x0, and linear at x > x0.
    '''
    y_pred = np.piecewise(x, [x <= x0, x > x0], [lambda x: y0, lambda x: k * (x - x0) + y0])
    return y_pred

# Function to derive the photometric error
def derive_phot_error() -> np.ndarray:
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

        # Reject the 'bad' data
        bad_data_indices = chisq > 0.1
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

    # Save the parameters as artifact
    np.save(XDATA_ARTIFACT_FILEPATH, x_data)
    np.save(YDATA_ARTIFACT_FILEPATH, y_data)
    pd.DataFrame(np.array([popt]), columns=['x0', 'y0', 'k']).to_csv(MODEL_ARTIFACT_FILEPATH, index=False)

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
    fig.savefig(IMG_OUTPUT_FILEPATH, dpi=300)
    
    return popt
    
def apply_phot_error(popt: np.ndarray) -> None:
    '''
    This function applies the error formula for every galaxy.
    '''
    for survey in SURVEY_LIST:
        df = pd.read_csv(INPUT_FILEPATH[survey])
        df['ei_j'] = piecewise_linear(np.array(df['j_m_ext'] - df['extinction_j']), *popt)
        df['er_j'] = 0.5 * df['ei_j']
        df = df[REQ_COLS[survey]].rename({'r_j': 'r', 'er_j': 'er', 's_scaled': 's', 'es_scaled': 'es', 'i_j': 'i', 'ei_j': 'ei'}, axis=1)
        df.to_csv(OUTPUT_FILEPATH[survey], index=False)

# def model_completeness(
#     surveys: List[str] = SURVEY_LIST,
#     markerstyles_: Dict[str, str] = DEFAULT_MARKERSTYLES,
#     lower_mag: float = MAG_LOW,
#     upper_mag: float = MAG_HIGH,
#     bin_width: float = COMPLETENESS_BIN_WIDTH,
#     p_val_reject: float = 0.01,
#     artifact_filepath: str = COMPLETENESS_ARTIFACT_PATH,
#     image_filepath: str = COMPLETENESS_IMAGE_PATH
# ):
#     model_params = []
#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
#     for survey in SURVEY_LIST:
#         filepath = OUTPUT_FILEPATH.get(survey)
#         df = pd.read_csv(filepath)

#         # Extinction-corrected magnitude
#         df['mag_j'] = df['j_m_ext'] - df['extinction_j']

#         # Create magnitude histogram
#         bins_ = np.arange(lower_mag, upper_mag + bin_width, bin_width)
#         labels_ = [i for i in range(1, len(bins_))]
#         df['mag_bin'] = pd.cut(df['mag_j'], bins_, labels=labels_)

#         # Count each bin
#         df_grouped = df[['mag_bin', 'mag_j']].groupby(by='mag_bin', observed=False).agg(
#             N=('mag_bin', 'count'),
#             mag_mean=('mag_j', 'mean')
#         )
#         df_grouped['log_N'] = np.log10(df_grouped['N'])
#         df_grouped = df_grouped[df_grouped['log_N'] > 0]

#         # Fit the function iteratively
#         x_data = df_grouped['mag_mean'].to_numpy()
#         y_data = df_grouped['log_N'].to_numpy()

#         # Remove outliers
# #         mask = x_data > 10.5
# #         x_data = x_data[mask][:-1]
# #         y_data = y_data[mask][:-1]

#         datacount = len(y_data)
#         is_converged = False
#         i = 1
#         while not is_converged:
#             # Fit the parameters
#             if survey != "LAMOST":
#                 # Fit linear+parabola model for 6dFGS and SDSS
#                 popt, pcov = curve_fit(completeness_linear_parabola, x_data, y_data, p0=[-4.5, 11.5, 5])
#             else:
#                 # Fit linear model for LAMOST
#                 popt, pcov = curve_fit(completeness_linear, x_data, y_data, p0=[-4.5])

#             # Calculate the predicted values and chi statistics
#             if survey != "LAMOST":
#                 y_pred = completeness_linear_parabola(x_data, *popt)
#             else:
#                 y_pred = completeness_linear(x_data, *popt)
#             chisq = ((y_data - y_pred) / y_pred)**2

#             # Reject the 'bad' data (chisq > 0.5)
#             bad_data_indices = chisq > p_val_reject
#             x_data = x_data[~bad_data_indices]
#             y_data = y_data[~bad_data_indices]
#             datacount_new = len(y_data)

#             is_converged = True if datacount == datacount_new else False
#             datacount = datacount_new
#             i += 1

#         # Save survey completeness parameter
#         model_params.append(popt)

#         # Create expected and fitted lines
#         if survey != "LAMOST":
#             df_grouped['N_model'] = df_grouped["mag_mean"].apply(lambda x: 10 ** completeness_linear_parabola(x, *popt))
#         else:
#             df_grouped['N_model'] = df_grouped["mag_mean"].apply(lambda x: 10 ** completeness_linear(x, *popt))

#         df_grouped["N_expected"] = 10 ** (0.6 * df_grouped["mag_mean"] + popt[0])
#         df_grouped["completeness"] = 100 * df_grouped["N"] / df_grouped["N_expected"]
#         df_grouped["completeness_model"] = 100 * df_grouped["N_model"] / df_grouped["N_expected"]

#         # First plot: log N vs magnitude
#         ax1.scatter(df_grouped['mag_mean'], df_grouped['log_N'], marker=markerstyles_[survey]) #, label=survey)
#         ax1.plot(df_grouped["mag_mean"], np.log10(df_grouped['N_model']), color='red')#, label='linear+parabola')

#         ax1.set_title(f"Differential count", fontsize=14)
#         ax1.set_xlabel(r'j_m_ext (mag)', fontsize=14)
#         ax1.set_ylabel(r'$\log N\ (\mathrm{deg}^{-2}\ \mathrm{mag}^{-1})$', fontsize=14)
#         # ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#         # ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#         # ax1.tick_params(axis='both', length=7.5, direction='in')
#         # ax1.tick_params(which='minor', length=2.5, direction='in')
#         # ax1.legend(fontsize=15)
#         ax1.grid(alpha=0.5, ls=':')

#         # Second plot: completeness (%) vs magnitude)
#         ax2.scatter(df_grouped["mag_mean"], df_grouped["completeness"], marker=markerstyles_[survey]) #, label=survey)
#         ax2.plot(df_grouped["mag_mean"], df_grouped["completeness_model"], color='red')
#         ax2.axhline(y=100., color='k', ls='--')

#         ax2.set_title(f"Magnitude completeness", fontsize=14)
#         ax2.set_xlabel(r'j_m_ext (mag)', fontsize=14)
#         ax2.set_ylabel(r'completeness (%)', fontsize=14)
#         # ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#         # ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#         # ax2.tick_params(axis='both', length=7.5, direction='in')
#         # ax2.tick_params(which='minor', length=2.5, direction='in')
#         # ax2.legend(fontsize=15)
#         ax2.grid(alpha=0.5, ls=':')
#         ax2.set_ylim(0, 140.)
        
#         plt.savefig(image_filepath, dpi=300)
        
#     pd.DataFrame(model_params, index=SURVEY_LIST, columns=['beta', 'x0', 'b']).to_csv(artifact_filepath)

# def calculate_completeness(mag, model_params):
#     N_expected = 10 ** (0.6 * mag + model_params[0])
#     N_model = 10 ** completeness_linear_parabola(mag, *model_params)
#     completeness = N_model / N_expected
#     return completeness

def combine_sdss_lamost() -> None:
    # Create empty dataframe
    df = pd.DataFrame()
    # Combine FP samples from SDSS and LAMOST into a single sample'
    for survey in ['SDSS', 'LAMOST']:
        filepath = OUTPUT_FILEPATH.get(survey)
        data = pd.read_csv(filepath)
        df = pd.concat([df, data])
    # # Drop duplicate measurements and keep the first (should be 6dFGS > SDSS > LAMOST)
    # count_ = len(df)
    # logger.info(f"Number of combined SDSS+LAMOST galaxies = {count_}.")
    # df = df.drop_duplicates(subset='tmass')
    # logger.info(f"Number of unique SDSS+LAMOST galaxies = {len(df)}. Galaxies dropped = {count_ - len(df)}.")

    df.to_csv(OUTPUT_FILEPATH.get('SDSS_LAMOST'), index=False)

def combine_6dfgs_sdss() -> None:
    # Create empty dataframe
    df = pd.DataFrame()
    # Combine FP samples from SDSS and LAMOST into a single sample'
    for survey in ['6dFGS', 'SDSS']:
        filepath = OUTPUT_FILEPATH.get(survey)
        data = pd.read_csv(filepath)
        df = pd.concat([df, data])
    # # Drop duplicate measurements and keep the first (should be 6dFGS > SDSS > LAMOST)
    # count_ = len(df)
    logger.info(f"Number of combined 6dFGS+SDSS galaxies = {len(df)}.")
    # df = df.drop_duplicates(subset='tmass')
    # logger.info(f"Number of unique 6dFGS+SDSS galaxies = {len(df)}. Galaxies dropped = {count_ - len(df)}.")

    df.to_csv(OUTPUT_FILEPATH.get('6dFGS_SDSS'), index=False)

def combine_all() -> None:
    # Create empty dataframe
    df = pd.DataFrame()
    # Combine FP samples of all survey into a single sample
    for survey in ['6dFGS', 'SDSS', 'LAMOST']:
        filepath = OUTPUT_FILEPATH.get(survey)
        data = pd.read_csv(filepath)
        df = pd.concat([df, data])
    # # Drop duplicate measurements and keep the first (should be 6dFGS > SDSS > LAMOST)
    # count_ = len(df)
    logger.info(f"Number of combined 6dF+SDSS+LAMOST galaxies = {len(df)}.")
    # df = df.drop_duplicates(subset='tmass')
    # logger.info(f"Number of unique 6dF+SDSS+LAMOST galaxies = {len(df)}. Galaxies dropped = {count_ - len(df)}.")

    df.to_csv(OUTPUT_FILEPATH.get('ALL_COMBINED'), index=False)

def main() -> None:
    try:
        logger.info(f"{'=' * 100}")
        logger.info(f"Deriving photometric errors. SMIN_SETTING = {SMIN_SETTING}.")
        popt = derive_phot_error()
        logger.info("Deriving photometric errors successful!")
        
        logger.info("Applying photometric errors and selecting the final columns...")
        apply_phot_error(popt)
        logger.info("Applying photometric errors and selecting the final columns successful!")
        
        # # Modelling the magnitude completeness
        # logger.info("Modelling magnitude completeness...")
        # model_completeness()
        # logger.info("Modelling magnitude completeness successful!")

        logger.info("Calculating completeness for each galaxy...")
        # Obtain completeness at each magnitude
        for survey in SURVEY_LIST:
            filepath = OUTPUT_FILEPATH.get(survey)
            df = pd.read_csv(filepath)

            # Set C_m = 1 by default
            df['C_m'] = 1
            # if survey in ['6dFGS', 'SDSS']:
            #     model_params = pd.read_csv(COMPLETENESS_ARTIFACT_PATH, index_col=0).loc[survey].to_numpy()
            #     df['C_m'] = (df['j_m_ext'] - df['extinction_j']).apply(lambda x: calculate_completeness(x, model_params))
            
            df.to_csv(filepath, index=False)
        logger.info("Calculating completeness for each galaxy successful!")
        
        # If veldisp cut is uniform, combine samples into a single sample (SDSS+LAMOST and 6dFGS+SDSS+LAMOST)
        if SMIN_SETTING==1:
            logger.info("Combining SDSS+LAMOST galaxies into a single sample...")
            combine_sdss_lamost()
            logger.info("Combining SDSS+LAMOST galaxies into a single sample successful!")

            logger.info("Combining 6dFGS+SDSS galaxies into a single sample...")
            combine_6dfgs_sdss()
            logger.info("Combining 6dFGS+SDSS galaxies into a single sample successful!")
            
            logger.info("Combining all galaxies into a single sample...")
            combine_all()
            logger.info("Combining all galaxies into a single sample successful!")

        logger.info('\n')

    except Exception as e:
        logger.error(f'Deriving photometric errors failed. Reason: {e}.')
    
if __name__ == '__main__':
    main()