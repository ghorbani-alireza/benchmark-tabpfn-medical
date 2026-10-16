import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import numpy as np
from sklearn.calibration import calibration_curve

from .config import PLOTS_PATH


# All plots use this order. QUADRO belongs with the classical methods,
# immediately after CatBoost. TabPFN versions stay together within foundations.
CLASSICAL_METHODS = ['L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'Quadro']
TABPFN_METHODS = ['TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35']
FOUNDATION_METHODS = TABPFN_METHODS + ['EXAONE', 'TabICL-V2']
METHOD_ORDER = CLASSICAL_METHODS + FOUNDATION_METHODS

COLOR_SCHEME = {
    'L-SLR':        '#2ca02c',
    'SLR':          '#ff7f0e',
    'RandomForest': '#d62728',
    'XGBoost':      '#9467bd',
    'CatBoost':     '#8c564b',
    'Quadro':       '#bcbd22',
    'TabPFN_v25':   '#c6dbef',
    'TabPFN_v26':   '#6baed6',
    'TabPFN_v3':    '#2171b5',
    'TabPFN_v35':   '#08306b',
    'EXAONE':       '#e377c2',
    'TabICL-V2':    '#000000',
}

# Only display labels change; saved result column names stay the same.
DISPLAY_NAMES = {
    'Quadro': 'QUADRO',
    'TabPFN_v25': 'TabPFN 2.5', 'TabPFN_v26': 'TabPFN 2.6',
    'TabPFN_v3': 'TabPFN 3', 'TabPFN_v35': 'TabPFN 3.5',
}


def _grouped_axis(ax, methods, frames=False):
    """Separate method groups with spacing and either labeled frames or light shading."""
    positions = np.arange(1, len(methods) + 1, dtype=float)
    classical_count = sum(method in CLASSICAL_METHODS for method in methods)
    tabpfn_count = sum(method in TABPFN_METHODS for method in methods)
    # Add a gap before foundations, and a smaller gap after the TabPFN subgroup.
    if classical_count and classical_count < len(methods):
        positions[classical_count:] += 0.8
    if tabpfn_count and classical_count + tabpfn_count < len(methods):
        positions[classical_count + tabpfn_count:] += 0.4

    # Keep both group borders and their labels inside the plotting area.
    outer_top = 0.96
    inner_top = 0.90
    # Put labels on the lower borders, with a separate level for TabPFN.
    height_points = max(ax.get_window_extent().height * 72 / ax.figure.dpi, 1)
    outer_bottom = max(0.025, 10 / height_points)
    inner_bottom = outer_bottom + 20 / height_points

    for title, members, background, border in [
        ('Classical methods', CLASSICAL_METHODS, '#eeeeee', '#a6611a'),
        ('Foundation models', FOUNDATION_METHODS, '#eaf2fa', '#2166ac'),
    ]:
        group_positions = [x for x, method in zip(positions, methods) if method in members]
        if not group_positions:
            continue
        left, right = min(group_positions) - 0.45, max(group_positions) + 0.45
        if frames:
            # A thin outline groups methods without adding colour behind the data.
            ax.add_patch(Rectangle(
                (left, outer_bottom), right - left, outer_top - outer_bottom,
                transform=ax.get_xaxis_transform(), fill=False,
                edgecolor=border, linewidth=1.1, clip_on=True, zorder=2,
            ))
            # Black text remains readable without relying on colour alone.
            ax.text((left + right) / 2, outer_bottom, title,
                    transform=ax.get_xaxis_transform(), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#222222', clip_on=True,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
        else:
            ax.axvspan(left, right, color=background, zorder=0)
            ax.annotate(title, xy=((left + right) / 2, 1), xycoords=ax.get_xaxis_transform(),
                        xytext=(0, 32), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', annotation_clip=False)

    tabpfn_positions = [x for x, method in zip(positions, methods) if method in TABPFN_METHODS]
    if tabpfn_positions:
        left, right = min(tabpfn_positions) - 0.39, max(tabpfn_positions) + 0.39
        if frames:
            # A thinner dashed frame shows the TabPFN subgroup inside foundations.
            ax.add_patch(Rectangle(
                (left, inner_bottom), right - left, inner_top - inner_bottom,
                transform=ax.get_xaxis_transform(), fill=False,
                edgecolor='#2166ac', linewidth=0.85, linestyle='--',
                clip_on=True, zorder=2,
            ))
            ax.text((left + right) / 2, inner_bottom, 'TabPFN',
                    transform=ax.get_xaxis_transform(), ha='center', va='center',
                    fontsize=9, color='#222222', clip_on=True,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
        else:
            ax.plot([left, left, right, right], [1.003, 1.01, 1.01, 1.003],
                    transform=ax.get_xaxis_transform(), color='#2171b5', clip_on=False)
            ax.annotate('TabPFN', xy=((left + right) / 2, 1), xycoords=ax.get_xaxis_transform(),
                        xytext=(0, 14), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, color='#2171b5', annotation_clip=False)
    return positions


def _pad_group_frames(ax):
    """Keep the plotted data inside the nested frames, including whiskers and outliers."""
    frames = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    if not frames:
        return
    bottom = max(frame.get_y() for frame in frames) + 0.015
    top = min(frame.get_y() + frame.get_height() for frame in frames) - 0.015
    data_low, data_high = ax.dataLim.intervaly
    low, high = ax.get_ylim()
    if not np.all(np.isfinite([data_low, data_high])):
        return
    # Expand the current limits enough to fit data between the innermost borders.
    span = max(high - low, (data_high - data_low) / (top - bottom),
               (data_high - low) / top, (high - data_low) / (1 - bottom))
    low = min(low, data_low - bottom * span)
    ax.set_ylim(low, low + span)


def _grouped_legend(fig, handles):
    """Center two method columns and return the top of the legend area."""
    sections = [
        ('Classical methods', CLASSICAL_METHODS),
        ('Foundation models', FOUNDATION_METHODS),
    ]
    legends = []
    bottom = 0.045 if 'Bayes ROC' in handles else 0.01
    for title, methods in sections:
        available = [method for method in methods if method in handles]
        if not available:
            continue
        legend = fig.legend(
            handles=[handles[method] for method in available],
            labels=[DISPLAY_NAMES.get(method, method) for method in available],
            title=title, loc='lower center', bbox_to_anchor=(0.5, bottom),
            ncol=1, fontsize=9, title_fontsize=10,
            frameon=False,
        )
        legend.get_title().set_fontweight('bold')
        legends.append(legend)

    if legends:
        # Measure the actual widths so missing methods still leave a centered legend.
        fig.draw_without_rendering()
        widths = [legend.get_window_extent().transformed(fig.transFigure.inverted()).width
                  for legend in legends]
        gap = 0.025
        left = (1 - sum(widths) - gap * (len(legends) - 1)) / 2
        for legend, width in zip(legends, widths):
            legend.set_bbox_to_anchor((left + width / 2, bottom))
            left += width + gap

    if 'Bayes ROC' in handles:
        legends.append(fig.legend(
            handles=[handles['Bayes ROC']], loc='lower center',
            bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=False,
        ))
    fig.draw_without_rendering()
    return max((legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
                for legend in legends), default=0)


def _title(setting, labels):
    """Use the supplied setting label, or the setting ID when no label is provided."""
    return labels.get(str(setting), str(setting)) if labels else str(setting)


# ─────────────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────────────
def plot_roc_curves_grid(errors_dict, settings_to_show, labels=None,
                         dir=PLOTS_PATH, figsize=(12, 10), save_path=None, *,
                         bayes_auc=None, show_bayes_roc=True):
    """Draw the average ROC curve for each model, with one panel per setting.
    
    Interpolate each repetition onto a common false-positive-rate grid before averaging.
    Supply distribution-based references as bayes_auc to add a dashed Bayes ROC.
    show_bayes_roc=False hides it. Monte Carlo curves estimate the population
    optimum; individual finite-test-sample curves can cross this reference.
    """
    color_scheme = COLOR_SCHEME
    models = METHOD_ORDER

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

        if show_bayes_roc and bayes_auc and setting in bayes_auc:
            reference = bayes_auc[setting]
            if all(key in reference for key in ('fpr', 'tpr')):
                line, = ax.plot(reference['fpr'], reference['tpr'], color='black',
                                linestyle='--', linewidth=2, label='Bayes ROC', zorder=4)
                legend_handles.setdefault('Bayes ROC', line)

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12)
        ax.grid(alpha=0.3)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    legend_top = _grouped_legend(fig, legend_handles)
    fig.suptitle('Average ROC Curves', fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, legend_top + 0.02, 1, 0.97])
    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Boxplots (MSE / AUC / Brier / any scalar metric)
# ─────────────────────────────────────────────────────────────────────
def plot_boxplots_grid(errors_dict, settings_to_show, metric='MSE', labels=None,
                       dir=PLOTS_PATH, figsize=(12, 10), save_path=None, *, group_frames=True,
                       bayes_auc=None, show_bayes_auc=True):
    """Compare a scalar measure across repetitions, with two setting panels per row.
    
    Group classical methods first, followed by foundation models and their TabPFN subgroup.
    Use group_frames=False for the earlier shaded-group style.
    Pass a setting-to-reference dictionary as bayes_auc for AUC plots.
    The reference line is on by default; show_bayes_auc=False hides it.
    """
    color_scheme = COLOR_SCHEME
    models = METHOD_ORDER

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

        data_to_plot, colors, shown_methods = [], [], []
        for model in models:
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                continue
            data_to_plot.append(df[col_name].dropna())
            colors.append(color_scheme[model])
            shown_methods.append(model)

        if not shown_methods:
            ax.set_visible(False)
            continue
        positions = _grouped_axis(ax, shown_methods, frames=group_frames)
        xtick_labels = [DISPLAY_NAMES.get(model, model) for model in shown_methods]
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # A population reference, not another fitted model or another box.
        if metric.upper() == 'AUC' and show_bayes_auc and bayes_auc and setting in bayes_auc:
            reference = bayes_auc[setting]
            value = reference['auc']
            prefix = 'Bayes AUC ≈' if reference.get('mc_se', 0) > 0 else 'Bayes AUC ='
            ax.axhline(value, color='black', linestyle='--', linewidth=1.3,
                       label=f'{prefix} {value:.3f}', zorder=4)
            # Place the label just above the line, aligned to the right edge.
            ax.annotate(f'{prefix} {value:.3f}', xy=(0.99, value),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 9),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                        zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12, pad=14 if group_frames else 54)
        ax.grid(axis='y', alpha=0.3)

        if metric.upper() == 'MSE':
            ax.set_ylim(0.0, 0.6)
        if group_frames:
            _pad_group_frames(ax)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()


def plot_boxplots_grid_lands(errors_dict, settings_to_show, metric='MSE', labels=None,
                             dir=PLOTS_PATH, figsize=(12, 10), save_path=None, *, group_frames=True,
                       bayes_auc=None, show_bayes_auc=True):
    """Draw grouped boxplots with four panels per row when there are at least four settings.

    Use group_frames=False for the earlier shaded-group style.
    Pass a setting-to-reference dictionary as bayes_auc for AUC plots.
    The reference line is on by default; show_bayes_auc=False hides it.
    """
    color_scheme = COLOR_SCHEME
    models = METHOD_ORDER

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

        data_to_plot, colors, shown_methods = [], [], []
        for model in models:
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                continue
            data_to_plot.append(df[col_name].dropna())
            colors.append(color_scheme[model])
            shown_methods.append(model)

        if not shown_methods:
            ax.set_visible(False)
            continue
        positions = _grouped_axis(ax, shown_methods, frames=group_frames)
        xtick_labels = [DISPLAY_NAMES.get(model, model) for model in shown_methods]
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # A population reference, not another fitted model or another box.
        if metric.upper() == 'AUC' and show_bayes_auc and bayes_auc and setting in bayes_auc:
            reference = bayes_auc[setting]
            value = reference['auc']
            prefix = 'Bayes AUC ≈' if reference.get('mc_se', 0) > 0 else 'Bayes AUC ='
            ax.axhline(value, color='black', linestyle='--', linewidth=1.3,
                       label=f'{prefix} {value:.3f}', zorder=4)
            # Place the label just above the line, aligned to the right edge.
            ax.annotate(f'{prefix} {value:.3f}', xy=(0.99, value),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 9),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                        zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(_title(setting, labels), pad=14 if group_frames else 54)
        ax.grid(axis='y', alpha=0.3, color='white')
        if group_frames:
            _pad_group_frames(ax)

    for idx in range(n_settings, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    if plt.get_backend().lower() != 'agg':
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────
def plot_calibration_grid(errors_dict, settings_to_show, labels=None, n_bins=10,
                          strategy='uniform', dir=PLOTS_PATH,
                          figsize=(12, 10), save_path=None):
    """Compare predicted probabilities with observed event frequencies.
    
    Pool the saved test predictions across repetitions and use the chosen probability bins.
    """
    color_scheme = COLOR_SCHEME
    models = METHOD_ORDER

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

    legend_top = _grouped_legend(fig, legend_handles)
    plt.tight_layout(rect=[0, legend_top + 0.02, 1, 1])

    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Brier vs AUC scatter
# ─────────────────────────────────────────────────────────────────────
def plot_brier_vs_auc_grid(errors_dict, settings_to_show, labels=None,
                           dir=PLOTS_PATH, figsize=(12, 10), save_path=None):
    """Plot each model's mean AUC against its mean Brier score, with one panel per setting."""
    color_scheme = COLOR_SCHEME
    models = METHOD_ORDER

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

    legend_top = _grouped_legend(fig, legend_handles)
    fig.suptitle('Brier vs. AUC', fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, legend_top + 0.02, 1, 0.97])

    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight')
    if plt.get_backend().lower() != 'agg':
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Runtime barplot
# ─────────────────────────────────────────────────────────────────────
def plot_runtime_barplot_simple(errors_dict, settings_to_show, labels=None,
                                dir=PLOTS_PATH, figsize=(10, 6), save_path=None):
    """Compare mean runtime per model, pooling saved timings from the selected settings."""
    color_scheme = COLOR_SCHEME
    methods = METHOD_ORDER

    runtime_data = {m: [] for m in methods}
    for setting in settings_to_show:
        df = errors_dict[setting]
        for m in methods:
            col = f'{m}_Time'
            if col in df.columns:
                runtime_data[m].extend(df[col].dropna().tolist())

    methods = [method for method in methods if runtime_data[method]]
    means = [np.mean(runtime_data[method]) for method in methods]

    fig, ax = plt.subplots(figsize=figsize)
    x = _grouped_axis(ax, methods)
    ax.bar(x, means, color=[color_scheme[m] for m in methods],
           edgecolor='black', linewidth=0.8, alpha=0.85)

    for i, mean in enumerate(means):
        if not np.isnan(mean):
            ax.text(x[i], mean, f'{mean:.2f}s', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods],
                       rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Average runtime (seconds)', fontsize=13)
    ax.set_title('Mean Runtime per Method', fontsize=14, pad=54)
    ax.margins(y=0.15)
    ax.grid(axis='y', alpha=0.3, color='white')
    plt.tight_layout()

    if save_path:
        plt.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    if plt.get_backend().lower() != 'agg':
        plt.show()