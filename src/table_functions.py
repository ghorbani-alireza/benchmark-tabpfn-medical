import os
import pandas as pd

from .config import (
    TABLES_PATH
)

#  one table, all metrics
def build_combined_table(summary_dfs, setting_names, filename,
                         metrics=None, errors_list=None, error_metrics=None):
    if metrics is None:
        metrics = ['AUC', 'MSE', 'Time']
    if error_metrics is None:
        error_metrics = []
    methods = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'TabPFN']
    rows = []

    for setting, df, err_df in zip(setting_names, summary_dfs, errors_list):
        for metric in metrics:
            row = {'Setting': setting, 'Metric': metric}
            if metric in error_metrics:
                # Read from errors_list
                for m in methods:
                    col = f'{m}_{metric}'
                    if col not in err_df.columns:
                        raise KeyError(f"Column '{col}' not found in errors_list for setting {setting}")
                    vals = err_df[col]
                    row[m] = f"{vals.mean():.3f} ± {vals.std():.3f}"
            else:
                mean_col = f"{metric}_Mean"
                std_col  = f"{metric}_Std"
                if mean_col not in df.columns:
                    mean_col = f"{metric} Mean"
                    std_col  = f"{metric} Std"
                for m in methods:
                    mean_val = df.loc[m, mean_col]
                    std_val  = df.loc[m, std_col]
                    row[m] = f"{mean_val:.3f} ± {std_val:.3f}"
            rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.set_index(['Setting', 'Metric'], inplace=True)


    csv_path = os.path.join(TABLES_PATH, filename)
    table_df.to_csv(csv_path)
    print(f"✓ Saved combined table to {csv_path}")

    def style_row(row):
        metric = row.name[1]
        num_vals = {}
        for col in row.index:
            try:
                num = float(row[col].split(' ± ')[0])
                num_vals[col] = num
            except:
                num_vals[col] = None
        if not any(num_vals.values()):
            return [''] * len(row)
        reverse = (metric == 'AUC')
        sorted_cols = sorted(num_vals.items(), key=lambda x: x[1], reverse=reverse)
        best = sorted_cols[0][0]
        second = sorted_cols[1][0] if len(sorted_cols) > 1 else None
        styles = []
        for col in row.index:
            if col == best:
                styles.append('text-decoration: underline')
            elif second is not None and col == second:
                styles.append('text-decoration: underline double')
            else:
                styles.append('')
        return styles

    return table_df.style.apply(style_row, axis=1)


################################################################################

# sensitivity gain table
def build_gain_table(base_dfs, sens_dfs, setting_names, filename, metric='AUC_Mean'):
    methods = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost','TabPFN']
    gain_data = []
    for base_df, sens_df, name in zip(base_dfs, sens_dfs, setting_names):
        delta = sens_df.loc[methods, metric] - base_df.loc[methods, metric]
        gain_data.append(delta)
    gain_df = pd.DataFrame(gain_data, index=setting_names, columns=methods)
    gain_df.index.name = 'Setting'

    csv_path = os.path.join(TABLES_PATH, filename)
    gain_df.to_csv(csv_path)
    print(f"✓ Saved gain table to {csv_path}")

    def highlight_gain(row):
        sorted_vals = row.sort_values(ascending=False)
        largest = sorted_vals.iloc[0]
        second = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None
        return [
            'text-decoration: underline' if val == largest else
            'text-decoration: underline double' if second is not None and val == second else ''
            for val in row
        ]

    styled = gain_df.style.apply(highlight_gain, axis=1).format(lambda x: f"{x:+.3f}")
    return styled


################################################################################

#  table for sensitivity analysis
def build_auc_brier_table(summary_dfs, errors_list, setting_names, filename, error_metrics=None):
    if error_metrics is None:
        error_metrics = ['Brier']

    methods = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'TabPFN']
    rows = []


    for metric in error_metrics:
        for name, err_df in zip(setting_names, errors_list):
            row_dict = {}
            for m in methods:
                col = f'{m}_{metric}'
                if col not in err_df.columns:
                    raise KeyError(f"Column '{col}' missing for setting {name}")
                vals = err_df[col]
                row_dict[m] = f"{vals.mean():.3f} ± {vals.std():.3f}"
            rows.append((name, metric, row_dict))


    for name, df in zip(setting_names, summary_dfs):
        auc_dict = {}
        # Find correct column names
        mean_col = 'AUC_Mean' if 'AUC_Mean' in df.columns else 'AUC Mean'
        std_col  = 'AUC_Std'  if 'AUC_Std'  in df.columns else 'AUC Std'
        if mean_col not in df.columns or std_col not in df.columns:
            raise KeyError(f"AUC columns missing in summary for {name}")
        for m in methods:
            auc_dict[m] = f"{df.loc[m, mean_col]:.3f} ± {df.loc[m, std_col]:.3f}"
        rows.append((name, 'ROC-AUC', auc_dict))


    index_tuples = []
    data = []
    for name in setting_names:
        # AUC first
        auc_row = next(r[2] for r in rows if r[0] == name and r[1] == 'ROC-AUC')
        index_tuples.append((name, 'ROC-AUC'))
        data.append([auc_row[m] for m in methods])
        # Then each error metric in the given order
        for metric in error_metrics:
            err_row = next(r[2] for r in rows if r[0] == name and r[1] == metric)
            index_tuples.append((name, metric))
            data.append([err_row[m] for m in methods])

    index = pd.MultiIndex.from_tuples(index_tuples, names=['Setting', 'Metric'])
    combined = pd.DataFrame(data, index=index, columns=methods)


    csv_path = os.path.join(TABLES_PATH, filename)
    combined.to_csv(csv_path)
    print(f"✓ Saved table to {csv_path}")

    # Styling
    def highlight(row):
        metric = row.name[1]
        num_vals = {}
        for col in row.index:
            try:
                num = float(row[col].split(' ± ')[0])
                num_vals[col] = num
            except:
                num_vals[col] = None
        if not any(num_vals.values()):
            return [''] * len(row)
        reverse = (metric == 'ROC-AUC')
        sorted_items = sorted(num_vals.items(), key=lambda x: x[1], reverse=reverse)
        best = sorted_items[0][0]
        second = sorted_items[1][0] if len(sorted_items) > 1 else None
        styles = []
        for col in row.index:
            if col == best:
                styles.append('text-decoration: underline')
            elif second is not None and col == second:
                styles.append('text-decoration: underline double')
            else:
                styles.append('')
        return styles

    return combined.style.apply(highlight, axis=1)