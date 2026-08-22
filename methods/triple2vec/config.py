import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
DATA_DIR = BASE_DIR + '/data/' 
SAMPLE_DIR = BASE_DIR + '/sample/'
MODEL_DIR = BASE_DIR + '/model/'
PARAM_DIR = BASE_DIR + '/output/param/'
OUTPUT_DIR = BASE_DIR + '/output/result/'

# Only create directories if explicitly enabled via env var
AUTO_CREATE_DIRS = os.getenv('RECSYS_AUTO_CREATE_DIRS', '0') == '1'
if AUTO_CREATE_DIRS:
    for DIR in [DATA_DIR, SAMPLE_DIR, MODEL_DIR, PARAM_DIR, OUTPUT_DIR]:
        if not os.path.exists(DIR):
            os.makedirs(DIR)