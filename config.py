import os 

# --- Project Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure input and output folders exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Local Dataset Paths ---
VARIABLES_2018 = os.path.join(DATA_DIR, 'variables_2018.csv')
VARIABLES_2021 = os.path.join(DATA_DIR, 'variables_2021.csv')
VARIABLES_2018_MDT_IND_EXT = os.path.join(DATA_DIR, 'Variables_MDT_IND_External_2018.csv')
VARIABLES_2021_MDT_IND_EXT = os.path.join(DATA_DIR, 'Variables_MDT_IND_External_2021.csv')