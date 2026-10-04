#src/config.py
from pathlib import Path

# src/config.py → parent.parent → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
IO_DIR            = PROJECT_ROOT / "io"

REAL_DATA_DIR     = IO_DIR / "real_data"
REAL_RESULTS_DIR  = IO_DIR / "real_data_results"
SIM_RESULTS_DIR   = IO_DIR / "sim_results"
TABLES_DIR        = IO_DIR / "tables"
PLOTS_DIR         = IO_DIR / "plots"

for _d in (REAL_DATA_DIR, REAL_RESULTS_DIR, SIM_RESULTS_DIR, TABLES_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Back-compat aliases for any code still using the old names
SIM_RESULTS_PATH  = str(SIM_RESULTS_DIR) + "/"
REAL_RESULTS_PATH = str(REAL_RESULTS_DIR) + "/"
REAL_DATA_PATH    = str(REAL_DATA_DIR) + "/"
TABLES_PATH       = str(TABLES_DIR) + "/"
PLOTS_PATH        = str(PLOTS_DIR) + "/"