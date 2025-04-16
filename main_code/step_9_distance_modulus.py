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

SMIN_SETTING = int(os.environ.get('SMIN_SETTING'))
FP_FIT_METHOD = int(os.environ.get('FP_FIT_METHOD'))

# Create boolean from FP_FIT_METHOD value
USE_FULL_FN = True if FP_FIT_METHOD == 0 else False

# Create logging instance
logger = get_logger('fit_logdist')

id_mapper = {
    "6dFGS": "_6dFGS",
    "SDSS": "objid",
    "LAMOST": "obsid"
}


def calculate_distance_modulus(
        z_dist_est: np.array,
        z_helio: np.array,
        logdist: np.array,
        logdist_err: np.array,
        H0: float = 100.0
) -> tuple[np.array, np.array]:
    """Function to calculate distance modulus and its corresponding error from log-distance ratio and its error

    Args:
        z_dist_est (np.array): redshift to estimate distance (CMB group redshift)
        z_helio (np.array): heliocentric redshift
        logdist (np.array): log-distance ratio
        logdist_err (np.array): error in log-distance ratio

    Returns:
        tuple[np.array, np.array]: tuple of distance modulus and its error
    """

    # Calculate luminosity distance (in Mpc)
    red_spline, lumred_spline, dist_spline, lumdist_spline, ez_spline = rz_table(H0=H0)
    d_C = sp.interpolate.splev(z_dist_est, dist_spline)
    d_L = (1 + z_helio) * d_C

    # Calculate distance modulus from logdist
    dist_mod = 5 * np.log10(d_L) - 5 * logdist + 25
    dist_mod_err = 5 * logdist_err

    return dist_mod, dist_mod_err


def calculate_group_average(
        df: pd.DataFrame,
        group_id_col: str,
        DM_src_col: str,
        eDM_src_col: str,
        avg_DM_dest_col: str,
        avg_eDM_dest_col: str,
) -> pd.DataFrame:
    """Calculate error-weighted distance modulus for every galaxy group.

    Args:
        df (pd.DataFrame): DataFrame containing group/cluster galaxies
        group_id_col (str): Group ID column name
        DM_src_col (str): distance modulus input column name
        eDM_src_col (str): error distance modulus input column name
        avg_DM_dest_col (str): averaged distance modulus output column name
        avg_eDM_dest_col (str): averaged error in distance modulus input column name

    Returns:
        pd.DataFrame: dataframe containing the averaged distance modulus and the error
    """

    # Calculate weight
    df["weight"] = 1 / df[eDM_src_col]**2

    # Calculate weight * DM
    df["weight_times_DM"] = df["weight"] * df[DM_src_col]

    # Group by Group ID and sum the weighted things
    df_grouped = df.groupby(by=group_id_col, observed=False).agg(
        numerator=("weight_times_DM", "sum"),
        denominator=("weight", "sum"),
    )

    # Calculate error-weighted average of distance modulus and its error
    df_grouped[avg_DM_dest_col] = df_grouped["numerator"] / df_grouped["denominator"]
    df_grouped[avg_eDM_dest_col] = 1 / np.sqrt(df_grouped["denominator"])

    df_grouped = df_grouped.reset_index()[[group_id_col, avg_DM_dest_col, avg_eDM_dest_col]]

    return df_grouped


def main():
    try:
        logger.info(f"{'=' * 50}")

        for survey in SURVEY_LIST:
            # Load data
            filepath = os.path.join(ROOT_PATH, f"data/foundation/logdist/smin_setting_1/fp_fit_method_0/{survey.lower()}.csv")
            df = pd.read_csv(filepath)

            # Create new columns for logdists from individual and combined FP
            df["logdist_individual_fp"] = df[f"logdist_{survey.lower()}"]
            df["logdist_err_individual_fp"] = df[f"logdist_err_{survey.lower()}"]

            df["logdist_combined_fp"] = df[f"logdist_all_combined"]
            df["logdist_err_combined_fp"] = df[f"logdist_err_all_combined"]

            # Calculate distance modulus using Logdist from individual FP
            logger.info("Calculating distance modulus using logdist from each survey's individual FP")
            df["DM_individual_fp"], df["eDM_individual_fp"] = calculate_distance_modulus(df["z_dist_est"].to_numpy(), df["zhelio"].to_numpy(), df[f"logdist_{survey.lower()}"].to_numpy(), df[f"logdist_err_{survey.lower()}"].to_numpy())

            logger.info("Calculating distance modulus using logdist from the combined FP")
            df["DM_combined_fp"], df["eDM_combined_fp"] = calculate_distance_modulus(df["z_dist_est"].to_numpy(), df["zhelio"].to_numpy(), df[f"logdist_all_combined"].to_numpy(), df[f"logdist_err_all_combined"].to_numpy())


            # Remove field galaxies
            df_group = df.copy()
            df_group = df_group[~(df_group["Group"].isin([-1, 0]))]

            for method in ["individual", "combined"]:
                # Calculate group-averaged distance modulus
                df_avg = calculate_group_average(
                    df=df_group,
                    group_id_col="Group",
                    DM_src_col=f"DM_{method}_fp",
                    eDM_src_col=f"eDM_{method}_fp",
                    avg_DM_dest_col=f"group_DM_{method}_fp",
                    avg_eDM_dest_col=f"group_eDM_{method}_fp"
                )

                # Join back to original data
                df = df.merge(df_avg, on="Group", how="left")

                # Use individual measurements for field galaxies
                df[f"group_DM_{method}_fp"] = df[f"group_DM_{method}_fp"].fillna(df[f"DM_{method}_fp"])
                df[f"group_eDM_{method}_fp"] = df[f"group_eDM_{method}_fp"].fillna(df[f"eDM_{method}_fp"])

            # Select relevant columns
            df = df[[
                'tmass', id_mapper[survey], 'ra', 'dec', 'zhelio', 'z_cmb', 'z_dist_est',
                'j_m_ext', 'extinction_j', 'kcor_j', 'r', 'er', 's', 'es', 'i', 'ei',
                'Group', 'Nr', 'logdist_individual_fp', 'logdist_err_individual_fp',
                'logdist_combined_fp', 'logdist_err_combined_fp', 'DM_individual_fp', 'eDM_individual_fp',
                'DM_combined_fp', 'eDM_combined_fp', 'group_DM_individual_fp',
                'group_eDM_individual_fp', 'group_DM_combined_fp', 'group_eDM_combined_fp'
            ]]

            # Save output data
            filepath = os.path.join(ROOT_PATH, f"data/foundation/distance_modulus/{survey.lower()}.csv")
            df.to_csv(filepath, index=False)
            
    except Exception as e:
        logger.error(f'Calculating distance modulus failed. Reason: {e}.')


if __name__ == "__main__":
    main()