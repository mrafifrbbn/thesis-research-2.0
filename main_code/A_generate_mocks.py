import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

from astropy.coordinates import SkyCoord

from jinja2 import Template

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get("ROOT_PATH")
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from src.filepaths import *
from src.utils.constants import *
from src.utils.logging_config import get_logger

logger = get_logger("generate_mocks")


def generate_extinction_data(input_filepath: str, output_filepath: str):
    """_summary_

    Args:
        input_filepath (str): path to the file 
        output_filepath (str): _description_
    """
    # Load recent data
    df = pd.read_csv(input_filepath)

    # Fetch relevant columns: 2MASS ID, RA, Dec, and J-band extinction
    df = df[['tmass', 'ra', 'dec', 'extinction_j']].rename(columns={'tmass':'obsid', 'extinction_j':'SF11_J'})

    # Calculate Galactic coordinates
    coords = SkyCoord(df['ra'], df['dec'], unit='deg', frame ='fk5')
    coords_gal = coords.transform_to('galactic')
    df['l'] = coords_gal.l.value
    df['b'] = coords_gal.b.value

    # Set other extinction values as 0
    df['SFD98_J'] = 0
    df['SFD98_H'] = 0
    df['SF11_H'] = 0
    df['SFD98_K'] = 0
    df['SF11_K'] = 0

    # Reorder the dataframe
    df = df[['obsid', 'ra', 'dec', 'l', 'b', 'SFD98_J', 'SF11_J', 'SFD98_H', 'SF11_H', 'SFD98_K', 'SF11_K']]
    df.to_csv(output_filepath, index=False, sep="\t")


def generate_config(
        survey: str,
        smin_setting: int,
        fp_fit_method: int,
        extinction_data_filepath: str,
        fp_fit_filepath: str,
        phot_error_filepath: str,
        final_fp_sample_filepath: str
        ) -> dict:
    """_summary_

    Args:
        survey (str): _description_
        smin_setting (int): _description_
        fp_fit_method (int): _description_
        extinction_data_filepath (str): _description_
        fp_fit_filepath (str): _description_
        phot_error_filepath (str): _description_
        final_fp_sample_filepath (str): _description_

    Returns:
        _type_: _description_
    """

    config = {}

    # Load final FP sample to obtain the number of galaxies and s error distributions
    df = pd.read_csv(final_fp_sample_filepath)
    ngals_mocks = len(df)
    mean_logds = np.log(df['es']).mean()
    std_logds = np.log(df['es']).std()

    # Load the best-fit FP
    df = pd.read_csv(fp_fit_filepath, index_col=[0])
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = df.loc[survey].to_numpy()

    # Load extinction data
    df = pd.read_csv(extinction_data_filepath)
    ngals_extinction_data = len(df)

    # Load photometric error data
    df = pd.read_csv(phot_error_filepath)
    mag_lim, err_constant, err_slope = df.loc[0].to_numpy()
    constant_pl = err_constant - err_slope * mag_lim

    # Generate output file name
    mock_output_name = os.path.join(ROOT_PATH, f"data/mocks/mock_galaxies/{survey.lower()}_mocks_smin_{smin_setting}_fp_fit_method_{fp_fit_method}.txt")

    config["SURVEY"] = str(survey)
    config["SMIN_SETTING"] = str(smin_setting)
    config["FP_FIT_METHOD"] = str(fp_fit_method)
    config["MAG_J_LIMIT"] = str(MAG_HIGH)
    config["SMIN"] = str(SURVEY_VELDISP_LIMIT[int(smin_setting)][survey])
    config["NFITS"] = str(1000)
    config["NGALS_MOCKS"] = str(ngals_mocks)
    config["MEAN_LOGDS"] = str(mean_logds)
    config["STD_LOGDS"] = str(std_logds)
    config["a_INPUT_VALUE"] = str(a)
    config["b_INPUT_VALUE"] = str(b)
    config["rmean_INPUT_VALUE"] = str(rmean)
    config["smean_INPUT_VALUE"] = str(smean)
    config["imean_INPUT_VALUE"] = str(imean)
    config["sigma1_INPUT_VALUE"] = str(sigma1)
    config["sigma2_INPUT_VALUE"] = str(sigma2)
    config["sigma3_INPUT_VALUE"] = str(sigma3)
    config["EXTINCTION_FILENAME"] = extinction_data_filepath
    config["NGALS_EXTINCTION_DATA"] = str(ngals_extinction_data)
    config["dI_MAG_SLOPE"] = str(2.5 * err_slope)
    config["dI_MAG_CONSTANT"] = str(2.5 * constant_pl)
    config["dI_ERR_CONSTANT"] = str(2.5 * err_constant)
    config["OUTPUT_FILENAME"] = mock_output_name

    return config

def generate_genrmockfp_file(
        template_filepath: str,
        template_render_result_filepath: str,
        config: dict
        ):
    
    # Open the templated CPP file
    with open(template_filepath, "r") as f:
        template_file = f.read()

    # Replace the templates using Jinja
    rendered_result = Template(template_file).render(config)

    # Save the rendered CPP file
    with open(template_render_result_filepath, "w") as f:
        f.write(rendered_result)

    return


def main():

    configs_list = []
    for survey in SURVEY_LIST + ["ALL_COMBINED"]:
        
        # Step 1: generate extinction data
        generate_extinction_data(
            input_filepath=FOUNDATION_ZONE_FP_SAMPLE_FILEPATHS[survey],
            output_filepath=EXTINCTION_DATA_FILEPATHS[survey]
        )

        # Step 2: generate the config
        for fp_fit_method in [0, 1]:

            # Define the filepaths dynamically
            fp_fit_filepath = os.path.join(ROOT_PATH, f'artifacts/fp_fit/smin_setting_{SMIN_SETTING}/fp_fit_method_{fp_fit_method}/fp_fits.csv')
            final_fp_sample_filepath = os.path.join(ROOT_PATH, f'data/foundation/fp_sample_final/smin_setting_{SMIN_SETTING}/fp_fit_method_{fp_fit_method}/{survey.lower()}.csv')

            config_ = generate_config(
                survey=survey,
                smin_setting=SMIN_SETTING,
                fp_fit_method=fp_fit_method,
                extinction_data_filepath=EXTINCTION_DATA_FILEPATHS[survey],
                fp_fit_filepath=fp_fit_filepath,
                phot_error_filepath=PHOT_ERROR_MODEL_FILEPATH,
                final_fp_sample_filepath=final_fp_sample_filepath
            )
            configs_list.append(config_)

    # Save the configs in a JSON file
    with open(MOCK_CONFIG_FILEPATH, "w") as json_file:
        json.dump(configs_list, json_file, indent=4)

    # Step 3: generate the mocks for every config
    for config in configs_list:
        # Generate the genrmockfp_cpp rendered file
        logger.info(f"Generating 1000 mocks @ {config['NGALS_MOCKS']} galaxies for {config['SURVEY']} with FP obtained from method {config['FP_FIT_METHOD']}")
        generate_genrmockfp_file(GENRMOCKFP_TEMPLATE_FILEPATH, GENRMOCKFP_CPP_FILEPATH, config)

        # Run the simulations
        GENRMOCKFP_FOLDER_PATH = Path(GENRMOCKFP_CPP_FILEPATH).parent
        subprocess.run(['make', 'clean'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)
        subprocess.run(['make', 'genrmockfp'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)
        subprocess.run(['./genrmockfp'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)

if __name__ == "__main__":
    main()