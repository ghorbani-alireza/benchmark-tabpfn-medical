import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from .config import PLOTS_PATH


# Single source of truth for colours and method order.
COLOR_SCHEME = {
    'L-SLR':        '#2ca02c',
    'SLR':          '#ff7f0e',
    'RandomForest': '#d62728',
    'XGBoost':      '#9467bd',
    'CatBoost':     '#8c564b',
    'TabPFN_v25':   '#c6dbef',
    'TabPFN_v26':   '#6baed6',
    'TabPFN_v3':    '#2171b5',
    'TabPFN_v35':   '#08306b',
    'EXAONE':       '#e377c2',
    'TabICL-V2':    '#000000',
}


def _title(setting, labels):
    return labels.get(str(setting), str(setting)) if labels else str(setting)


# ─────────────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────────────
def plot_roc_curves_grid(errors_dict, settings_to_show, labels=None,
                         dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    color_scheme = COLOR_SCHEME
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n_settings == 1 else axes.flatten()

    base_fpr = np.linspace(0, 1, 100)
    legend_handles = {}

    for idx, setting in enumerate(settings_to_show):
        ax = axes[idx]
        df = errors_dict[setting]

        for method in models:
            fpr_col = f'{method}_FPR'
            tpr_col = f'{method}_TPR'
            if fpr_col not in df.columns or tpr_col not in df.columns:
                continue

            tprs_interp = []
            for fpr, tpr in zip(df[fpr_col].tolist(), df[tpr_col].tolist()):
                if len(fpr) == 0 or len(tpr) == 0:
                    continue
                tprs_interp.append(np.interp(base_fpr, fpr, tpr))
            if not tprs_interp:
                continue

            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_auc = np.mean(df[f'{method}_AUC'])

            line, = ax.plot(base_fpr, mean_tpr, linewidth=2,
                            color=color_scheme[method],
                            label=f"{method} (AUC={mean_auc:.3f})")
            legend_handles.setdefault(method, line)

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12)
        ax.grid(alpha=0.3)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles=list(legend_handles.values()),
               labels=list(legend_handles.keys()),
               loc='lower center', bbox_to_anchor=(0.5, 0.03),
               ncol=len(legend_handles), fontsize=10, frameon=True)

    plt.suptitle('Average ROC Curves', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Boxplots (MSE / AUC / Brier / any scalar metric)
# ─────────────────────────────────────────────────────────────────────
def plot_boxplots_grid(errors_dict, settings_to_show, metric='MSE', labels=None,
                       dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    color_scheme = COLOR_SCHEME
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n_settings == 1 else axes.flatten()

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        data_to_plot, positions, colors, xtick_labels = [], [], [], []
        for i, model in enumerate(models):
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                continue
            data_to_plot.append(df[col_name].dropna())
            positions.append(i + 1)
            colors.append(color_scheme[model])
            xtick_labels.append(model)

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        if metric.upper() == 'MSE':
            ax.set_ylim(0.0, 0.6)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_boxplots_grid_lands(errors_dict, settings_to_show, metric='MSE', labels=None,
                             dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    color_scheme = COLOR_SCHEME
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 4 if n_settings >= 4 else 1
    nrows = (n_settings + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n_settings == 1 else axes.flatten()

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        data_to_plot, positions, colors, xtick_labels = [], [], [], []
        for i, model in enumerate(models):
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                continue
            data_to_plot.append(df[col_name].dropna())
            positions.append(i + 1)
            colors.append(color_scheme[model])
            xtick_labels.append(model)

        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(_title(setting, labels))
        ax.grid(axis='y', alpha=0.3, color='white')

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
def plot_calibration_grid(errors_dict, settings_to_show, labels=None, n_bins=10,
                          strategy='uniform', dir=PLOTS_PATH,
                          figsize=(12, 10), save_path=None):
    color_scheme = COLOR_SCHEME
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n_settings == 1 else axes.flatten()

    legend_handles = {}

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        y_all = np.concatenate(df['y_test_all'].tolist())

        for model in models:
            proba_col = f'{model}_Probas'
            if proba_col not in df.columns:
                continue
            proba_all = np.concatenate(df[proba_col].tolist())
            prob_true, prob_pred = calibration_curve(
                y_all, proba_all, n_bins=n_bins, strategy=strategy)
            line, = ax.plot(prob_pred, prob_true, marker='o', linewidth=2,
                            color=color_scheme[model], label=model)
            legend_handles.setdefault(model, line)

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12)
        ax.grid(alpha=0.3)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles=list(legend_handles.values()),
               labels=list(legend_handles.keys()),
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_handles), fontsize=10, frameon=True)

    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Brier vs AUC scatter
# ─────────────────────────────────────────────────────────────────────
def plot_brier_vs_auc_grid(errors_dict, settings_to_show, labels=None,
                           dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    color_scheme = COLOR_SCHEME
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if n_settings == 1 else axes.flatten()

    legend_handles = {}

    for idx, setting in enumerate(settings_to_show):
        ax = axes[idx]
        df = errors_dict[setting]

        for method in models:
            auc_col = f'{method}_AUC'
            brier_col = f'{method}_Brier'
            if auc_col in df.columns and brier_col in df.columns:
                scatter = ax.scatter(df[auc_col].mean(), df[brier_col].mean(),
                                     label=method, c=color_scheme[method],
                                     s=120, edgecolors='black',
                                     linewidth=1.5, alpha=0.9)
                legend_handles.setdefault(method, scatter)

        ax.set_xlabel('Mean AUC')
        ax.set_ylabel('Mean Brier Score')
        ax.set_title(_title(setting, labels))
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.0, 0.5)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle('Brier vs. AUC', fontsize=14, y=0.98)
    fig.legend(handles=list(legend_handles.values()),
               labels=list(legend_handles.keys()),
               loc='upper center', bbox_to_anchor=(0.5, 0.96),
               ncol=len(legend_handles), fontsize=10, frameon=True)

    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Runtime barplot
# ─────────────────────────────────────────────────────────────────────
def plot_runtime_barplot_simple(errors_dict, settings_to_show, labels=None,
                                dir=PLOTS_PATH, figsize=(10, 6), save_path=None):
    color_scheme = COLOR_SCHEME
    methods = list(color_scheme.keys())

    runtime_data = {m: [] for m in methods}
    for setting in settings_to_show:
        df = errors_dict[setting]
        for m in methods:
            col = f'{m}_Time'
            if col in df.columns:
                runtime_data[m].extend(df[col].dropna().tolist())

    means = [np.mean(runtime_data[m]) if runtime_data[m] else np.nan
             for m in methods]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(methods))
    ax.bar(x, means, color=[color_scheme[m] for m in methods],
           edgecolor='black', linewidth=0.8, alpha=0.85)

    for i, mean in enumerate(means):
        if not np.isnan(mean):
            ax.text(i, mean, f'{mean:.2f}s', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=12)
    ax.set_ylabel('Average runtime (seconds)', fontsize=13)
    ax.set_title('Mean Runtime per Method', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, color='white')
    plt.tight_layout()

    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.show()