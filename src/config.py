# config.py
MAIN_PATH = None
REAL_DATA_PATH = None
SIM_RESULTS_PATH = None
TABLES_PATH = None
PLOTS_PATH = None


def set_paths(main_path):
    global MAIN_PATH, REAL_DATA_PATH, SIM_RESULTS_PATH, TABLES_PATH, PLOTS_PATH
    MAIN_PATH = main_path
    REAL_DATA_PATH = MAIN_PATH + "io/real_data/"
    SIM_RESULTS_PATH = MAIN_PATH + "io/sim_results/"
    TABLES_PATH = MAIN_PATH + "io/tables/"
    PLOTS_PATH = MAIN_PATH + "io/plots/"


# Settings
USE_TABPFN_CLIENT = False # Flase: local; True: client
#two option for tabpfn
# clinet or local - client is GPU but limited
# https://ux.priorlabs.ai