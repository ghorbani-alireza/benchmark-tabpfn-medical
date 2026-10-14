import os, sys
import numpy as np
import glob, re
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import logging
from pathlib import Path
import time
import warnings
import pandas as pd
import json

from . import config

# string lists into python lists
def _parse_array(s):
    """Read full JSON arrays, or complete arrays from older result files."""
    if not isinstance(s, str):
        return s
    try:
        return np.asarray(json.loads(s))
    except json.JSONDecodeError:
        if "..." in s:
            raise ValueError("Saved array is truncated; missing values cannot be recovered.")
        return np.fromstring(s.strip().strip("[]").replace(",", " "), sep=" ")


def fix_list_columns(df):
    """Restore saved prediction and ROC arrays from text columns in a results DataFrame."""
    list_keys = ('_FPR', '_TPR', '_Probas', 'y_test_all')
    for col in df.columns:
        if any(k in col for k in list_keys):
            df[col] = df[col].apply(_parse_array)
    return df

    
def load_fixed(files):
    """Read the selected result CSV files and restore their array columns."""
    out = {}
    for sid, path in files.items():
        df = pd.read_csv(path)
        out[sid] = fix_list_columns(df)
    return out

#################################################################################

# metadata for synthetic
def make_metadata(n, mu, dist, iter):
    """Build a short text description of a synthetic experiment for display."""
    return (f"n_tr={n['n1_tr']}/{n['n2_tr']} |"
           f"n_te={n['n1_te']}/{n['n2_te']} |"
           f"dim={len(mu['mu1'])} |"
           f"distribution={dist} |"
           f"iterations={iter}")
# metadata for real data
def make_real_metadata(dataset_name, n, iter):
    """Build a short text description of a real-data experiment for display."""
    return (f"Dataset: {dataset_name} | "
            f"n_tr={n['n1_tr']}/{n['n2_tr']} | "
            f"n_te={n['n1_te']}/{n['n2_te']} | "
            f"iterations={iter}")




#################################################################################

# function to save outputs
import json
import os
import pandas as pd

from . import config


def save_outputs(errors, name, output_dir=None, meta=None):
    """Save full results to CSV and, when supplied, experiment metadata to JSON.
    
    Prediction and ROC arrays are written as complete JSON lists inside CSV cells.
    """
    if output_dir is None:
        output_dir = config.SIM_RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)

    # ── main data ──
    csv_path = os.path.join(output_dir, f"{name}.csv")
    frame = pd.DataFrame(errors)
    for column in frame.columns:
        if column.endswith(("_FPR", "_TPR", "_Probas")) or column == "y_test_all":
            frame[column] = frame[column].map(lambda values: json.dumps(np.asarray(values).tolist()))
    frame.to_csv(csv_path, index=False)

    # ── sidecar metadata ──
    if meta is not None:
        meta_path = os.path.join(output_dir, f"{name}.meta.json")

        def _to_jsonable(v):
            import numpy as np
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            return v

        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items()}
            if isinstance(d, (list, tuple)):
                return [_clean(x) for x in d]
            return _to_jsonable(d)

        with open(meta_path, "w") as f:
            json.dump(_clean(meta), f, indent=2)

    return csv_path


#################################################################################

# Keep model messages in files while the main progress bar stays visible.

@contextmanager
def capture_output(warnings_file, output_dir):
    """Temporarily send model messages to files instead of the notebook/terminal.

    The analysis uses this as:
        with capture_output(warnings_file, output_dir) as (warns, progress):
            # Run the models; pass file=progress to the main tqdm bar.

    Python warnings are saved as JSON. Printed messages and library logs go
    to the companion .log file. The original output settings are restored
    when the block ends, including when a model raises an error.
    """
    # Choose a unique warning filename unless the caller supplied one.
    if warnings_file is None:
        warnings_file = Path(output_dir) / "warnings" / f"{time.time_ns()}.warnings.json"
    warning_path = Path(warnings_file)
    warning_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original screen output for the main progress bar.
    progress = sys.stderr
    saved_warnings = []
    # Library logging may use existing screen handlers, so redirect those too.
    loggers = [logging.getLogger(name) for name in (
        "", "exaonetabular", "huggingface_hub", "tabicl", "tabpfn",
        "tabpfn_v25", "tabpfn_v26", "tabpfn_v3", "tabpfn_v35",
    )]
    # Remember the current settings so this changes only the analysis block.
    original_logging = [(logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
    # "x" creates a new file and prevents overwriting an earlier run.
    with warning_path.open("x", encoding="utf-8") as warning_file:
        with warning_path.with_suffix(".log").open("x", encoding="utf-8") as log:
            handler = logging.StreamHandler(log)
            try:
                for logger in loggers:
                    logger.handlers = [handler]
                    logger.setLevel(logging.INFO)
                    logger.propagate = False
                # Capture once around the whole run, including its thread workers.
                # Capture print() and library progress messages, including TabICL.
                with redirect_stdout(log), redirect_stderr(log):
                    with warnings.catch_warnings(record=True) as recorded:
                        # Record every warning, including repeated convergence warnings.
                        warnings.simplefilter("always")
                        try:
                            # Run the simulation/resampling code inside its with block.
                            yield saved_warnings, progress
                        finally:
                            # Convert warning objects into ordinary dictionaries for JSON.
                            saved_warnings.extend({
                                "category": item.category.__name__,
                                "message": str(item.message),
                                "filename": item.filename,
                                "lineno": item.lineno,
                            } for item in recorded)
            finally:
                # Restore logging and save warnings even if execution failed.
                for logger, (handlers, level, propagate) in zip(loggers, original_logging):
                    logger.handlers = handlers
                    logger.setLevel(level)
                    logger.propagate = propagate
                json.dump(saved_warnings, warning_file, indent=2)


#################################################################################
#finding and reading saved results
def _natural_key(name):
    """Split names into text and numbers so setting 2 sorts before setting 10."""
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', name)]


def discover_sim(sim_dir, variant='baseline'):
    """Find baseline or sensitivity result CSV files in the selected directory."""
    pattern = ('simulation_setting*_sensitivity.csv' if variant == 'sensitivity'
               else 'simulation_setting*.csv')
    full = os.path.join(glob.escape(sim_dir), pattern)
    files = sorted(glob.glob(full),
                   key=lambda p: _natural_key(os.path.basename(p)))

    out = {}
    for f in files:
        base = os.path.basename(f).replace('.csv', '')
        if variant == 'baseline' and base.endswith('_sensitivity'):
            continue
        sid = base.replace('simulation_setting', '').replace('_sensitivity', '')
        out[f'S{sid}'] = f
    return out


def discover_real(real_dir):
    """Find real-data result CSV files in the selected directory."""
    full = os.path.join(glob.escape(real_dir), 'realdata_*.csv')
    files = sorted(glob.glob(full),
                   key=lambda p: _natural_key(os.path.basename(p)))
    return {f'D{i}': f for i, f in enumerate(files, 1)}

def read_labels(files):
    """Read plot labels from the metadata files stored beside result CSV files."""
    import json
    out = {}
    for sid, path in files.items():
        meta_path = path.replace('.csv', '.meta.json')
        label = sid
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                m = json.load(fh)
            label = m.get('label') or m.get('dataset') or sid
        out[sid] = label
    return out
    
    


def read_bayes_auc(files):
    """Load Bayes references beside CSVs, or calculate from saved distributions.

    Older results without full distribution metadata and real-data results are
    skipped: their generating distributions cannot safely be inferred.
    This reads metadata only; it does not fit models or modify saved results.
    """
    from .settings import calculate_bayes_auc
    references = {}
    for name, path in files.items():
        meta_path = Path(path).with_suffix('.meta.json')
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('bayes_auc') is not None:
            references[name] = meta['bayes_auc']
        elif all(key in meta for key in ('mu', 'sigma1', 'sigma2', 'dist')):
            references[name] = calculate_bayes_auc(meta)
    return references
