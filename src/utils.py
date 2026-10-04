import os, sys
import numpy as np
import glob, re
import logging
from contextlib import contextmanager
import warnings
import pandas as pd
import json
from sklearn.exceptions import ConvergenceWarning

from . import config

# string lists into python lists
def _parse_array(s):
    """'[0.1 0.2 0.3]' or '[0.1, 0.2, 0.3]' → np.array([0.1, 0.2, 0.3])."""
    if not isinstance(s, str):
        return s                      # already an array/list
    s = s.strip().strip('[]')
    if not s:
        return np.array([])
    return np.fromstring(s.replace(',', ' '), sep=' ')


def fix_list_columns(df):
    """Parse FPR / TPR / Probas / y_test_all columns from strings to arrays."""
    list_keys = ('_FPR', '_TPR', '_Probas', 'y_test_all')
    for col in df.columns:
        if any(k in col for k in list_keys):
            df[col] = df[col].apply(_parse_array)
    return df

    
def load_fixed(files):
    out = {}
    for sid, path in files.items():
        df = pd.read_csv(path)
        out[sid] = fix_list_columns(df)
    return out

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

# function to save outputs
import json
import os
import pandas as pd

from . import config


def save_outputs(errors, name, output_dir=None, meta=None):
    """Save a result dict as <name>.csv plus <name>.meta.json.
    """
    if output_dir is None:
        output_dir = config.SIM_RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)

    # ── main data ──
    csv_path = os.path.join(output_dir, f"{name}.csv")
    pd.DataFrame(errors).to_csv(csv_path, index=False)
    print(f"✓ Saved {csv_path}")

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
        print(f"✓ Saved {meta_path}")

    return csv_path


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
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# tabpfn logging output
logging.getLogger('tabpfn').setLevel(logging.ERROR)

#################################################################################
#finding and reading saved results
def _natural_key(name):
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', name)]


def discover_sim(sim_dir, variant='baseline'):
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
    full = os.path.join(glob.escape(real_dir), 'realdata_*.csv')
    files = sorted(glob.glob(full),
                   key=lambda p: _natural_key(os.path.basename(p)))
    return {f'D{i}': f for i, f in enumerate(files, 1)}

def read_labels(files):
    """{sid: path} → {sid: label} from .meta.json. Falls back to sid."""
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
    
    
