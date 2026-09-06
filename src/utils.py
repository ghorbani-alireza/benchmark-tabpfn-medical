import os
import logging
import sys
import re
from contextlib import contextmanager
import warnings
import gc
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from . import config


# fix multiple DataFrames at once
def fix_all_errors_datasets(*dataframes):
    return tuple(fix_list_columns(df.copy()) for df in dataframes)


# string lists into python lists
def fix_list_columns(df):
    for col in df.columns:
        # columns wuth list‑like data
        if any(key in col for key in ['_FPR', '_TPR', '_Probas', 'y_test_all']):
            def convert(cell):
                # only on strings
                if isinstance(cell, str):
                    cell_fixed = re.sub(r'(\d+\.?\d*)\s+', r'\1,', cell.strip())
                    # remove trailing comma
                    cell_fixed = cell_fixed.rstrip(',')
                return cell
            df[col] = df[col].apply(convert)
    return df

#################################################################################

# metadata for synthetic
def make_metadata(n, mu, dist, iter):
    return (f"n_tr={n['n1_tr']}/{n['n2_tr']} |"
           f"n_te={n['n1_te']}/{n['n2_te']} |"
           f"dim={len(mu['mu1'])} |"
           f"distribution={dist} |"
           f"iterations={iter}")
# metadata for real data
def make_real_metadata(dataset_name, n, iter):
    return (f"Dataset: {dataset_name} | "
            f"n_tr={n['n1_tr']}/{n['n2_tr']} | "
            f"n_te={n['n1_te']}/{n['n2_te']} | "
            f"iterations={iter}")


#################################################################################

# Summary Table
def create_summary(errors, setting_name="", metadata=None):

    methods = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'TabPFN']

    summary_dict = {}

    for method in methods:
        method_data = {}

        # AUC
        auc_key = f"{method}_AUC"
        if auc_key in errors and errors[auc_key]:
            method_data['AUC_Mean'] = np.mean(errors[auc_key])
            method_data['AUC_Std'] = np.std(errors[auc_key])
        else:
            method_data['AUC_Mean'] = np.nan
            method_data['AUC_Std'] = np.nan

        # MSE
        mse_key = f"{method}_MSE"
        if mse_key in errors and errors[mse_key]:
            method_data['MSE_Mean'] = np.mean(errors[mse_key])
            method_data['MSE_Std'] = np.std(errors[mse_key])
        else:
            method_data['MSE_Mean'] = np.nan
            method_data['MSE_Std'] = np.nan

        # Time
        time_key = f"{method}_Time"
        if time_key in errors and errors[time_key]:
            method_data['Time_Mean'] = np.mean(errors[time_key])
            method_data['Time_Std'] = np.std(errors[time_key])
        else:
            method_data['Time_Mean'] = np.nan
            method_data['Time_Std'] = np.nan

        summary_dict[method] = method_data
    summary_df = pd.DataFrame(summary_dict).T
    summary_df.attrs['metadata'] = metadata
    summary_df = summary_df.sort_values('AUC_Mean', ascending=False) # sort by auc

    return summary_df


#################################################################################

# function to save simulation
def save_simulation_outputs(errors, captured_warnings, setting_name, n, mu, dist, iter_,
                            errors_filename=None, summary_filename=None, warnings_filename=None):

    # default filenames
    if errors_filename is None:
        errors_filename = f"{setting_name.lower().replace(' ', '_')}_errors.csv"
    if summary_filename is None:
        summary_filename = f"{setting_name.lower().replace(' ', '_')}_summary.csv"
    if warnings_filename is None:
        warnings_filename = f"{setting_name.lower().replace(' ', '_')}_warnings.csv"


    # errors
    errors_copy = errors.copy()
    errors_df = pd.DataFrame(errors_copy)
    errors_df.to_csv(config.SIM_RESULTS_PATH + errors_filename, index=True)
    print(f"✓ Errors saved to {errors_filename}")

    # free memory
    del errors_copy
    del errors_df
    gc.collect()

    # warnings if any
    if captured_warnings:
        warnings_df = pd.DataFrame(captured_warnings)
        warnings_df.to_csv(config.SIM_RESULTS_PATH + warnings_filename, index=True)
        print(f"✓ Warnings saved to {warnings_filename} (total: {len(captured_warnings)})")
        del warnings_df
    else:
        print(f"✓ No warnings were captured")

    # summary
    summary_df = create_summary(errors, setting_name,
                                metadata=make_metadata(n, mu, dist, iter_))
    summary_df.to_csv(config.SIM_RESULTS_PATH + summary_filename, index=True)
    print(f"✓ Summary saved to {summary_filename}")

    # memory
    del summary_df
    del errors
    gc.collect()

#################################################################################


# save real data outputs - the same as above
def save_real_data_outputs(errors, dataset_name, n, iter_,
                           errors_filename=None, summary_filename=None, warnings_filename=None):

    if errors_filename is None:
        errors_filename = f"{dataset_name.lower().replace(' ', '_')}_errors.csv"
    if summary_filename is None:
        summary_filename = f"{dataset_name.lower().replace(' ', '_')}_summary.csv"
    if warnings_filename is None:
        warnings_filename = f"{dataset_name.lower().replace(' ', '_')}_warnings.csv"

    # errors
    errors_copy = errors.copy()
    errors_df = pd.DataFrame(errors_copy)
    errors_df.to_csv(config.REAL_DATA_PATH + errors_filename, index=True)
    print(f"✓ Errors saved to {errors_filename}")

    del errors_copy, errors_df
    gc.collect()

    # summary
    metadata = make_real_metadata(dataset_name, n, iter_)
    summary_df = create_summary(errors, setting_name=dataset_name, metadata=metadata)
    summary_df.to_csv(config.REAL_DATA_PATH + summary_filename, index=True)
    print(f"✓ Summary saved to {summary_filename}")

    del summary_df, errors
    gc.collect()


#################################################################################

# there are useless warnings & messages interrupting tqdm - >:(

@contextmanager
def suppress_output():
    # save
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull

        # disable tqdm
        old_tqdm_disable = os.environ.get('TQDM_DISABLE', None)
        os.environ['TQDM_DISABLE'] = '1'

        # tabPFN verbosity
        old_tabpfn_verbose = os.environ.get('TABPFN_VERBOSE', None)
        os.environ['TABPFN_VERBOSE'] = '0'

        try:
            yield
        finally:
            # Restore
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            if old_tqdm_disable is None:
                os.environ.pop('TQDM_DISABLE', None)
            else:
                os.environ['TQDM_DISABLE'] = old_tqdm_disable

            if old_tabpfn_verbose is None:
                os.environ.pop('TABPFN_VERBOSE', None)
            else:
                os.environ['TABPFN_VERBOSE'] = old_tabpfn_verbose


current_iteration = 0
captured_warnings = []

def capture_warnings(message, category, filename, lineno, file=None, line=None):
    captured_warnings.append({
        'iteration': current_iteration,
        'message': str(message),
        'category': category.__name__,
        'filename': filename,
        'lineno': lineno
    })

# convergenceWarning display
#warnings.filterwarnings('ignore', category=ConvergenceWarning)

# tabpfn logging output
logging.getLogger('tabpfn').setLevel(logging.ERROR)