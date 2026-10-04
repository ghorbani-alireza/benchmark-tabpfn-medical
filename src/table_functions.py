import os
import numpy as np
import pandas as pd

from .config import TABLES_PATH

METHODS = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost',
           'TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35',
           'EXAONE', 'TabICL-V2']

# Scalar metrics only. FPR/TPR/Probas are list-valued — used by plots.
METRICS = ['AUC', 'MSE', 'Brier', 'Time', 'CalSlope', 'CalIntercept']

HIGHER_IS_BETTER = {'AUC'}


def build_measures_table(errors_dict, metrics=None, filename='table.csv',
                         methods=None, tables_dir=None):
    """rows=(Setting, Metric), cols=method, values='mean ± std'.

    errors_dict : {setting_id: pd.DataFrame}
    """
    if not errors_dict:
        print(f"⚠️ no data to build {filename} — skipping")
        return None
    
    metrics = metrics or METRICS
    methods = methods or METHODS
    tables_dir = tables_dir or TABLES_PATH

    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            row = {'Setting': sid, 'Metric': metric}
            for m in methods:
                col = f'{m}_{metric}'
                if col in df.columns:
                    v = pd.to_numeric(df[col], errors='coerce').dropna()
                    row[m] = f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else '—'
                else:
                    row[m] = '—'
            rows.append(row)

    table = pd.DataFrame(rows).set_index(['Setting', 'Metric'])
    table.to_csv(os.path.join(tables_dir, filename))
    print(f"✓ {filename}")
    return table.style.apply(_style_best, axis=1)


def build_gain_table(base_dict, sens_dict, metric='AUC',
                     filename='gains.csv', methods=None, tables_dir=None):
    """Sensitivity mean − baseline mean, per (setting, method)."""
    methods = methods or METHODS
    tables_dir = tables_dir or TABLES_PATH
    
    if not base_dict or not sens_dict:
        print(f"⚠️ empty input for {filename} — skipping")
        return None

    common = [s for s in base_dict if s in sens_dict]

    rows = []
    for sid in common:
        row = {'Setting': sid}
        b, s = base_dict[sid], sens_dict[sid]
        for m in methods:
            col = f'{m}_{metric}'
            bv = (pd.to_numeric(b[col], errors='coerce').dropna().mean()
                  if col in b.columns else np.nan)
            sv = (pd.to_numeric(s[col], errors='coerce').dropna().mean()
                  if col in s.columns else np.nan)
            row[m] = sv - bv
        rows.append(row)

    table = pd.DataFrame(rows).set_index('Setting')
    table.to_csv(os.path.join(tables_dir, filename))
    print(f"✓ {filename}")
    return table.style.apply(_style_gain, axis=1).format(lambda x: f"{x:+.3f}")


def _style_best(row):
    metric = row.name[1]
    reverse = metric in HIGHER_IS_BETTER
    vals = {}
    for col in row.index:
        try:
            vals[col] = float(row[col].split(' ± ')[0])
        except Exception:
            pass
    if not vals:
        return [''] * len(row)
    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=reverse)
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return [
        'text-decoration: underline' if c == best
        else 'text-decoration: underline double' if c == second
        else ''
        for c in row.index
    ]


def _style_gain(row):
    ordered = row.sort_values(ascending=False)
    top = ordered.iloc[0]
    second = ordered.iloc[1] if len(ordered) > 1 else None
    return [
        'text-decoration: underline' if v == top
        else 'text-decoration: underline double' if second is not None and v == second
        else ''
        for v in row
    ]