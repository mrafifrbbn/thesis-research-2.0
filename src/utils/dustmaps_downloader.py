import os
from dustmaps.config import config
from dotenv import load_dotenv
load_dotenv()

DUSTMAPS_CONFIG_PATH = os.environ.get('DUSTMAPS_CONFIG_PATH')

config['data_dir'] = DUSTMAPS_CONFIG_PATH

if __name__ == '__main__':
    import dustmaps.sfd
    dustmaps.sfd.fetch()
