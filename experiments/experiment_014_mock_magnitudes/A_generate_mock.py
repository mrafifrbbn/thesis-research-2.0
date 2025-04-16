import os
import sys
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

ROOT_PATH = os.environ.get('ROOT_PATH')
if not ROOT_PATH in sys.path: sys.path.append(ROOT_PATH)

from main_code.filepaths import GENRMOCKFP_TEMPLATE_FILEPATH, GENRMOCKFP_CPP_FILEPATH
from main_code.A_generate_mocks import generate_genrmockfp_file

def main():

    # Load config file
    filepath = "/Users/mrafifrbbn/Documents/thesis/thesis-research-2.0/experiments/experiment_014_mock_magnitudes/mock_config.json"
    with open(filepath, 'r') as f:
        cfg = json.load(f)

    # Replace template to create genrmockfp file
    generate_genrmockfp_file(GENRMOCKFP_TEMPLATE_FILEPATH, GENRMOCKFP_CPP_FILEPATH, cfg)

    # Run the simulations
    GENRMOCKFP_FOLDER_PATH = Path(GENRMOCKFP_CPP_FILEPATH).parent
    subprocess.run(['make', 'clean'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)
    subprocess.run(['make', 'genrmockfp'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)
    subprocess.run(['./genrmockfp'], cwd=GENRMOCKFP_FOLDER_PATH, capture_output=True, text=True)

if __name__ == "__main__":
    main()