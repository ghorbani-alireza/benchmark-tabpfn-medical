import os
import numpy as np
import pandas as pd

from .config import TABLES_PATH
from .plot_functions import CLASSICAL_METHODS, FOUNDATION_METHODS

METHODS = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost',
           'TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35',
           'EXAONE', 'TabICL-V1', 'TabICL-V2', 'QDA', 'Quadro']

# Scalar metrics only. FPR/TPR/Probas are list-valued — used by plots.
METRICS = ['AUC', 'MSE', 'Brier', 'Time', 'SetupTime', 'TuneTime',
           'FitTime', 'PredictTime', 'CalSlope', 'CalIntercept']

HIGHER_IS_BETTER = {'AUC'}


def build_measures_table(errors_dict, metrics=None, filename='table.csv',
                         methods=None, tables_dir=None):
    """rows=(Setting, Metric), cols=method, values='mean ± std'.

    errors_dict : {setting_id: pd.DataFrame}
    Set filename=None to display the styled table without saving a CSV.
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
    if filename is not None:
        table.to_csv(os.path.join(tables_dir, filename))
        print(f"✓ {filename}")
    return table.style.apply(_style_best, axis=1)


def build_gain_table(base_dict, sens_dict, metric='AUC',
                     filename='gains.csv', methods=None, tables_dir=None, metrics=None):
    """Sensitivity mean minus baseline mean for matching settings and measures.

    These are differences of means: baseline and sensitivity runs are independent.
    Pass metrics=METRICS for a table of all scalar measures.
    """
    methods = methods or METHODS
    tables_dir = tables_dir or TABLES_PATH
    measures = metrics if metrics is not None else [metric]

    if not base_dict or not sens_dict:
        print(f"⚠️ empty input for {filename} — skipping")
        return None

    common = [s for s in base_dict if s in sens_dict]
    if not common:
        print(f"⚠️ no matching settings for {filename} — skipping")
        return None

    rows = []
    for sid in common:
        b, s = base_dict[sid], sens_dict[sid]
        for measure in measures:
            row = {'Setting': sid, 'Metric': measure}
            for m in methods:
                col = f'{m}_{measure}'
                bv = (pd.to_numeric(b[col], errors='coerce').replace([np.inf, -np.inf], np.nan).mean()
                      if col in b.columns else np.nan)
                sv = (pd.to_numeric(s[col], errors='coerce').replace([np.inf, -np.inf], np.nan).mean()
                      if col in s.columns else np.nan)
                row[m] = sv - bv
            rows.append(row)

    table = pd.DataFrame(rows)
    if metrics is None:
        table = table.drop(columns='Metric').set_index('Setting')
    else:
        table = table.set_index(['Setting', 'Metric'])
    if filename is not None:
        table.to_csv(os.path.join(tables_dir, filename))
        print(f"✓ {filename}")
    return table.style.apply(_style_gain, axis=1, metric=metric).format('{:+.3f}', na_rep='—')


def _rank_methods(df, methods, metric):
    """Rank observed means; calibration uses mean absolute distance from its ideal."""
    scores = {}
    for method in methods:
        column = f'{method}_{metric}'
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if metric == 'CalSlope':
            values = (values - 1.0).abs()
        elif metric == 'CalIntercept':
            values = values.abs()
        if values.notna().any():
            scores[method] = values.mean()
    return sorted(scores, key=scores.get, reverse=metric in HIGHER_IS_BETTER)


def build_best_methods_table(errors_dict, metrics=None, filename='best_methods.csv', tables_dir=None):
    """Best and second-best methods in each group, per setting/dataset and measure.

    Rank once using all repetitions; show the raw measure as mean and sample SD.
    Ties use the existing method order. This is a descriptive result summary.
    """
    metrics = metrics or METRICS
    tables_dir = tables_dir or TABLES_PATH
    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            for group, methods in [('Classical', CLASSICAL_METHODS), ('Foundation', FOUNDATION_METHODS)]:
                ranked = _rank_methods(df, methods, metric)
                if not ranked:
                    continue
                row = {'Setting': sid, 'Metric': metric, 'Group': group}
                for rank, label in [(0, 'Best'), (1, 'Second')]:
                    if rank < len(ranked):
                        method = ranked[rank]
                        values = pd.to_numeric(df[f'{method}_{metric}'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                        row[f'{label} method'] = method
                        row[f'{label} mean'] = values.mean()
                        row[f'{label} SD'] = values.std()
                        row[f'{label} runs'] = values.count()
                    else:
                        row[f'{label} method'] = '—'
                        row[f'{label} mean'] = np.nan
                        row[f'{label} SD'] = np.nan
                        row[f'{label} runs'] = 0
                rows.append(row)
    if not rows:
        print(f"⚠️ no methods to summarize for {filename} — skipping")
        return None
    table = pd.DataFrame(rows).set_index(['Setting', 'Metric', 'Group'])
    if filename is not None:
        table.to_csv(os.path.join(tables_dir, filename))
        print(f"✓ {filename}")
    return table.style.format({column: '{:.3f}' for column in table if column.endswith(('mean', 'SD'))}, na_rep='—')


def build_effect_table(errors_dict, metrics=None, filename='best_vs_best.csv', tables_dir=None, all_models=False):
    """Summarize within-run differences, foundation minus classical, as mean and SD.

    Select each group's best method once per measure using all repetitions.
    With all_models=True, compare every model with the best method in the other
    group. Calibration effects use raw slopes/intercepts, not their distances.
    """
    metrics = metrics or METRICS
    tables_dir = tables_dir or TABLES_PATH
    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            classical = _rank_methods(df, CLASSICAL_METHODS, metric)
            foundation = _rank_methods(df, FOUNDATION_METHODS, metric)
            if not classical or not foundation:
                continue
            pairs = [(foundation[0], classical[0])]
            if all_models:
                pairs += [(method, classical[0]) for method in foundation[1:]]
                pairs += [(foundation[0], method) for method in classical[1:]]
            for foundation_method, classical_method in pairs:
                foundation_values = pd.to_numeric(df[f'{foundation_method}_{metric}'], errors='coerce')
                classical_values = pd.to_numeric(df[f'{classical_method}_{metric}'], errors='coerce')
                paired = pd.DataFrame({'Foundation': foundation_values, 'Classical': classical_values})
                paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
                differences = paired['Foundation'] - paired['Classical']
                rows.append({
                    'Setting': sid, 'Metric': metric,
                    'Classical method': classical_method, 'Foundation method': foundation_method,
                    'Classical rank': classical.index(classical_method) + 1,
                    'Foundation rank': foundation.index(foundation_method) + 1,
                    'Mean effect': differences.mean(), 'SD effect': differences.std(),
                    'Paired runs': len(differences),
                })
    if not rows:
        print(f"⚠️ no group comparisons for {filename} — skipping")
        return None
    table = pd.DataFrame(rows).set_index(['Setting', 'Metric'])
    if filename is not None:
        table.to_csv(os.path.join(tables_dir, filename))
        print(f"✓ {filename}")
    return table.style.format({'Mean effect': '{:+.3f}', 'SD effect': '{:.3f}'}, na_rep='—')


def _style_best(row):
    metric = row.name[1]
    reverse = metric in HIGHER_IS_BETTER
    vals = {}
    for col in row.index:
        try:
            value = float(row[col].split(' ± ')[0])
            if metric == 'CalSlope':
                value = abs(value - 1.0)
            elif metric == 'CalIntercept':
                value = abs(value)
            vals[col] = value
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


def _style_gain(row, metric='AUC'):
    if isinstance(row.name, tuple):
        metric = row.name[1]
    # A raw calibration change has no single better direction.
    if metric in ('CalSlope', 'CalIntercept'):
        return [''] * len(row)
    ordered = row.dropna().sort_values(ascending=metric not in HIGHER_IS_BETTER)
    if ordered.empty:
        return [''] * len(row)
    top = ordered.iloc[0]
    second = ordered.iloc[1] if len(ordered) > 1 else None
    return [
        'text-decoration: underline' if v == top
        else 'text-decoration: underline double' if second is not None and v == second
        else ''
        for v in row
    ]