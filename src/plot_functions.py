import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from .config import (
    PLOTS_PATH
)

# Plot for ROC curves
def plot_roc_curves_grid(errors_dict, settings_to_show, dir=PLOTS_PATH,
                         figsize=(12, 10), save_path=None):
    # color scheme
    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_settings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Common FPR grid
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


            fpr_list = df[fpr_col].tolist()
            tpr_list = df[tpr_col].tolist()


            tprs_interp = []
            for fpr, tpr in zip(fpr_list, tpr_list):

                if len(fpr) == 0 or len(tpr) == 0:
                    continue
                tpr_interp = np.interp(base_fpr, fpr, tpr)
                tprs_interp.append(tpr_interp)
            if not tprs_interp:
                continue
            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_auc = np.mean(df[f'{method}_AUC'])

            # ROC curve
            line, = ax.plot(base_fpr, mean_tpr, linewidth=2,
                            color=color_scheme[method],
                            label=f"{method} (AUC={mean_auc:.3f})")
            if method not in legend_handles:
                legend_handles[method] = line

        # diagonal line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Chance')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(setting, fontsize=16)
        ax.grid(alpha=0.3)

    # unused subplots hide
    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    # single legend and location
    fig.legend(handles=list(legend_handles.values()),
               labels=[f"{m}" for m in legend_handles.keys()],
               loc='lower center', bbox_to_anchor=(0.5, 0.03),
               ncol=len(models), fontsize=10, frameon=True)

    plt.suptitle('Average ROC Curves – All Methods (100 iterations)', fontsize=14, y=1.02)
    plt.tight_layout()
    # adjust legend
    plt.subplots_adjust(bottom=0.08)
    if save_path:
        plt.savefig(dir+save_path, dpi=300, bbox_inches='tight')
    plt.show()

################################################################################

# Boxplot for MSE
def plot_boxplots_grid(errors_dict, settings_to_show, metric='MSE',
                       dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    # descriptive titles mapped
    setting_title_map = {
        "Setting 1": "Setting 1: Small-sample, low-dimensional Gaussian",
        "Setting 2": "Setting 2: Nonlinear interactions with no mean shift",
        "Setting 3": "Setting 3: Nonlinear interactions with sparse linear signal",
        "Setting 4": "Setting 4: Class imbalance",
        "Setting 5": "Setting 5: Heavy-tailed elliptical distributions",
        "Setting 6": "Setting 6: Heavy-tailed data with local correlation",
        "Setting 7": "Setting 7: Sparse Gaussian (p=40)",
        "Setting 8": "Setting 8: High-dimensional Sparse Gaussian (p=100)",
        1: "Setting 1: Small-sample, low-dimensional Gaussian",
        2: "Setting 2: Nonlinear interactions with no mean shift",
        3: "Setting 3: Nonlinear interactions with sparse linear signal",
        4: "Setting 4: Class imbalance",
        5: "Setting 5: Heavy-tailed elliptical distributions",
        6: "Setting 6: Heavy-tailed data with local correlation",
        7: "Setting 7: Sparse Gaussian (p=40)",
        8: "Setting 8: High-dimensional Sparse Gaussian (p=100)",
    }

    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_settings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        # Collect data for each method
        data_to_plot = []
        positions = []
        colors = []
        for i, model in enumerate(models):
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                print(f"Warning: {col_name} not found for {setting}, skipping {model}")
                continue
            values = df[col_name].dropna()
            data_to_plot.append(values)
            positions.append(i + 1)
            colors.append(color_scheme[model])


        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels([m for m in models if f'{m}_{metric}' in df.columns],
                           rotation=25, ha='right', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)


        title_key = setting if isinstance(setting, str) else setting
        full_title = setting_title_map.get(title_key, str(setting))
        ax.set_title(full_title, fontsize=14)  # Reduced fontsize slightly to fit longer text

        ax.grid(axis='y', alpha=0.3)

        #  [0, 1]
        ax.set_ylim(0.42, 1)

    # hide unused subplots
    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight')
    plt.show()

################################################################################

# for workshop
def plot_boxplots_grid_lands(errors_dict, settings_to_show, metric='MSE',
                       dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 4 if n_settings >= 4 else 1
    nrows = (n_settings + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)



    if n_settings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        data_to_plot = []
        positions = []
        colors = []
        for i, model in enumerate(models):
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                print(f"Warning: {col_name} not found for {setting}, skipping {model}")
                continue
            values = df[col_name].dropna()
            data_to_plot.append(values)
            positions.append(i + 1)
            colors.append(color_scheme[model])


        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels([m for m in models if f'{m}_{metric}' in df.columns],
                           rotation=25, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'{setting}')
        ax.grid(axis='y', alpha=0.3, color='white')  # white grid stands out gently on green


    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(dir+save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

################################################################################

# calibration plot
def plot_calibration_grid(errors_dict, settings_to_show, n_bins=10, strategy='uniform',
                          dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    # color scheme
    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_settings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()


    legend_handles = {}

    for idx, setting in enumerate(settings_to_show):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df = errors_dict[setting]

        y_test_list = df['y_test_all'].tolist()
        y_all = np.concatenate(y_test_list)

        for model in models:
            proba_col = f'{model}_Probas'
            if proba_col not in df.columns:
                print(f"Warning: {proba_col} not found for {setting}, skipping {model}")
                continue
            proba_list = df[proba_col].tolist()
            proba_all = np.concatenate(proba_list)
            prob_true, prob_pred = calibration_curve(y_all, proba_all, n_bins=n_bins, strategy=strategy)
            line, = ax.plot(prob_pred, prob_true, marker='o',  linewidth=2, #markersize=2,
                            color=color_scheme[model], label=model)
            if model not in legend_handles:
                legend_handles[model] = line


        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability', fontsize=14)
        ax.set_ylabel('Fraction of Positives', fontsize=14)
        ax.set_title(f'{setting}', fontsize=16)#Calibration –
        ax.grid(alpha=0.3)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    # single legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.legend(handles=list(legend_handles.values()),
               labels=list(legend_handles.keys()),
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(models), fontsize=10, frameon=True)

    #plt.suptitle('Calibration Curves – All Methods (100 iterations)', fontsize=14, y=1.02)

    if save_path:
        plt.savefig(dir+save_path, dpi=300, bbox_inches='tight')
    plt.show()

################################################################################

# Scatter plot for AUC vs. Brier score
def plot_brier_vs_auc_grid(errors_dict, settings_to_show, dir=PLOTS_PATH,
                           figsize=(12, 10), save_path=None):
    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    models = list(color_scheme.keys())

    n_settings = len(settings_to_show)
    ncols = 2 if n_settings >= 2 else 1
    nrows = (n_settings + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_settings == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    legend_handles = {}

    for idx, setting in enumerate(settings_to_show):
        ax = axes[idx]
        df = errors_dict[setting]

        for method in models:
            auc_col = f'{method}_AUC'
            brier_col = f'{method}_Brier'
            if auc_col in df.columns and brier_col in df.columns:
                auc_mean = df[auc_col].mean()
                brier_mean = df[brier_col].mean()
                scatter = ax.scatter(auc_mean, brier_mean,
                                     label=method, c=color_scheme[method],
                                     s=120, edgecolors='black', linewidth=1.5, alpha=0.9)
                if method not in legend_handles:
                    legend_handles[method] = scatter

        ax.set_xlabel('Mean AUC')
        ax.set_ylabel('Mean Brier Score')
        ax.set_title(setting)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.04, 0.21)

    # unused subplots
    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)


    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.suptitle('Brier vs. AUC – Each point is a method (mean over iterations)', fontsize=14, y=0.98)

    fig.legend(handles=list(legend_handles.values()),
               labels=list(legend_handles.keys()),
               loc='upper center', bbox_to_anchor=(0.5, 0.96),  # inside figure coordinates
               ncol=len(models), fontsize=10, frameon=True)

    if save_path:
        plt.savefig(dir+save_path, dpi=300, bbox_inches='tight')
    plt.show()

################################################################################


def plot_runtime_barplot_simple(errors_dict, settings_to_show,
                                dir=PLOTS_PATH, figsize=(8, 6), save_path=None):
    color_scheme = {
        'SLR': '#ff7f0e',
        'L-SLR': '#2ca02c',
        'RandomForest': '#d62728',
        'XGBoost': '#9467bd',
        'CatBoost': '#8c564b',
        'TabPFN': '#1f77b4'
    }
    methods = ['SLR','L-SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'TabPFN']


    runtime_data = {m: [] for m in methods}
    for setting in settings_to_show:
        df = errors_dict[setting]
        for m in methods:
            col = f'{m}_Time'
            if col in df.columns:
                runtime_data[m].extend(df[col].dropna().tolist())

    means = [np.mean(runtime_data[m]) for m in methods]


    fig, ax = plt.subplots(figsize=figsize)
    #fig.patch.set_facecolor('#E2F0D9')
    #ax.set_facecolor('#E2F0D9')

    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=[color_scheme[m] for m in methods],
                  edgecolor='black', linewidth=0.8, alpha=0.85)


    for i, (m, mean) in enumerate(zip(methods, means)):
        ax.text(i, mean + 0.1, f'{mean:.2f}s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=12)
    ax.set_ylabel('Average runtime (seconds)', fontsize=13)
    ax.set_title('Mean Runtime per Method (across all settings)', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, color='white')
    plt.tight_layout()

    if save_path:
        plt.savefig(dir + save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    plt.show()