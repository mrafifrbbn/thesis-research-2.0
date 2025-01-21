import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from dotenv import load_dotenv
load_dotenv(override=True)

root_dir = os.environ.get("ROOT_PATH")
if not root_dir in sys.path: sys.path.append(root_dir)

from src.utils.constants import *
from src.utils.CosmoFunc import *

from src.filepaths import *
from src.step_7_fit_fp import fit_FP
from src.step_8_fit_logdist import fit_logdist

# Absolute path to the parent directory of this file
file_root_dir = Path(__file__).parent


def split_richness(survey: str, df: DataFrame) -> None:
    richness_bins = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 99999)]

    for i, richness_bin in enumerate(richness_bins):
        lower_limit, upper_limit = richness_bin

        df_ = df.copy()

        # Filter based on richness
        df_ = df_[(df_["Nr"] >= lower_limit) & (df_["Nr"] < upper_limit)]
        df_["richness_lower_limit"] = lower_limit
        df_["richness_upper_limit"] = upper_limit

        # Save the dataframe
        save_path = os.path.join(file_root_dir, f"fp_richness_bin/")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_.to_csv(os.path.join(save_path, f"{survey.lower()}_{i}.csv"), index=False)

    return


def combine_richness(folder_path: str, survey: str) -> None:

    df = pd.DataFrame()
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        df_temp = pd.read_csv(filepath)
        df = pd.concat([df, df_temp], axis=0)

    df.to_csv(os.path.join(file_root_dir, f"{survey.lower()}_final.csv"), index=False)
    
    return


def main():
    try:
        # Load desired survey
        for survey in ["SDSS"]:
            df = pd.read_csv(os.path.join(root_dir, f"data/foundation/fp_sample/smin_setting_1/{survey.lower()}.csv"))

            # Split the data based on richness
            split_richness(survey, df)

            # Load each richness bin
            params_richness = []
            richness_bin_dir = os.path.join(file_root_dir, "fp_richness_bin")
            for file in os.listdir(richness_bin_dir):
                file_no = (file.split("_")[-1]).split(".")[0]
                if file.startswith(survey.lower()) and file.endswith(".csv"):
                    filepath = os.path.join(richness_bin_dir, file)
                    df = pd.read_csv(filepath)

                    # Fit the FP
                    params, df_fitted = fit_FP(
                        survey=survey,
                        df=df,
                        smin=SURVEY_VELDISP_LIMIT[1]['6dFGS'],
                        param_boundaries=[(1.4, 2.5), (-1.1, -0.7), (-0.2, 0.4), (2.1, 2.4), (3.1, 3.5), (0.0, 0.06), (0.20, 0.45), (0.1, 0.25)],
                        reject_outliers=True,
                        use_full_fn=True,
                    )

                    # Derive log-distance ratios
                    df_fitted = fit_logdist(
                        survey=survey,
                        df=df,
                        smin=SURVEY_VELDISP_LIMIT[1]['6dFGS'],
                        FPlabel=survey.lower(),
                        FPparams=params,
                        use_full_fn=True,
                        save_posterior=False
                    )
                    df_fitted.to_csv(os.path.join(file_root_dir, f"logdist/{file}"), index=False)

                    params = list(params)
                    params.append(file_no)
                    params.append(len(df))
                    params.append(df["richness_lower_limit"].values.tolist()[0])
                    params.append(df["richness_upper_limit"].values.tolist()[0])

                    params_richness.append(params)

            df_fitted_fp = pd.DataFrame(params_richness, columns=["a", "b", "rmean", "smean", "imean", "sigma1", "sigma2", "sigma3", "richness_bin", "N_data", "richness_lower_limit", "richness_upper_limit"])
            df_fitted_fp.to_csv(os.path.join(file_root_dir, f"{survey.lower()}_fp_fit.csv"), index=False)

            combine_richness(os.path.join(file_root_dir, "logdist"), survey)
    except Exception as e:
        print(f"Fitting FP & deriving log-distance ratios by richness for {survey} failed. Reason: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()