"""Run simulations, sensitivity analysis, resampling, or report saved results."""

import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=['simulation', 'sensitivity', 'resampling', 'report'])
    parser.add_argument('--repetitions', type=int, help='Number of Monte Carlo repetitions')
    parser.add_argument('--seed', type=int, help='Random seed; default 2025 (sensitivity: 2026)')
    parser.add_argument('--output', type=Path, help='New directory for this analysis')
    parser.add_argument('--results', type=Path, help='Saved result directory to report')
    parser.add_argument('--no-bayes-auc', action='store_true', help='Hide the Bayes AUC line in reports')
    args = parser.parse_args()
    if args.analysis == 'report':
        if args.results is None:
            parser.error('report requires --results')
    elif args.repetitions is None or args.repetitions < 1:
        parser.error('choose a positive --repetitions count')

    # Use a plotting backend that saves figures without opening windows.
    import matplotlib
    matplotlib.use('Agg')
    from benchmark import SETTINGS, SENSITIVITY, simulation, evaluate_on_real_data, save_outputs, calculate_bayes_auc
    from benchmark import load_echo_data, load_glucose_data, load_blood_gas_data
    from benchmark import config

    # Report existing results without loading data or fitting models.
    if args.analysis == 'report':
        from benchmark.utils import load_fixed, read_labels, read_bayes_auc
        from benchmark.table_functions import build_measures_table
        from benchmark.plot_functions import (
            plot_boxplots_grid, plot_boxplots_grid_lands, plot_roc_curves_grid,
            plot_calibration_grid, plot_brier_vs_auc_grid, plot_runtime_barplot_simple,
        )
        files = {path.stem: path for path in sorted(args.results.glob('*.csv'))}
        if not files:
            parser.error('no result CSV files found in --results directory')
        results = load_fixed(files)
        labels = read_labels({name: str(path) for name, path in files.items()})
        bayes = read_bayes_auc(files) if not args.no_bayes_auc else {}
        names = list(results)
        table_dir = args.results / 'tables'
        plot_dir = args.results / 'plots'
        table_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        build_measures_table(results, filename='measures.csv', tables_dir=table_dir)
        plots = str(plot_dir) + '/'
        plot_boxplots_grid(results, names, labels=labels, metric='AUC', dir=plots, save_path='auc_box.png', bayes_auc=bayes, show_bayes_auc=not args.no_bayes_auc)
        plot_boxplots_grid_lands(results, names, labels=labels, metric='AUC', dir=plots, save_path='auc_box_landscape.png', bayes_auc=bayes, show_bayes_auc=not args.no_bayes_auc)
        plot_roc_curves_grid(results, names, labels=labels, dir=plots, save_path='roc.png')
        plot_calibration_grid(results, names, labels=labels, dir=plots, save_path='calibration.png')
        plot_brier_vs_auc_grid(results, names, labels=labels, dir=plots, save_path='brier_auc.png')
        plot_runtime_barplot_simple(results, names, labels=labels, dir=plots, save_path='runtime.png')
        print(f'Tables and plots saved in {args.results}')
        return

    # Make a new result folder so a previous analysis is not overwritten.
    parent = config.REAL_RESULTS_DIR if args.analysis == 'resampling' else config.SIM_RESULTS_DIR
    output = args.output or parent / f'{args.analysis}-{time.time_ns()}'
    if output.exists():
        parser.error('output directory already exists; choose another --output name')
    output.mkdir(parents=True)
    seed = args.seed if args.seed is not None else (2026 if args.analysis == 'sensitivity' else 2025)

    # Main and sensitivity simulations use the same direct simulation function.
    if args.analysis in ('simulation', 'sensitivity'):
        settings = SENSITIVITY if args.analysis == 'sensitivity' else SETTINGS
        for setting_id, cfg in settings.items():
            suffix = '_sensitivity' if args.analysis == 'sensitivity' else ''
            name = f'simulation_setting{setting_id}{suffix}'
            errors, warns = simulation(
                n=cfg['n'], mu=cfg['mu'], sigma1=cfg['sigma1'], sigma2=cfg['sigma2'],
                dist=cfg['dist'], df=cfg.get('df'), iter=args.repetitions, g_seed=seed,
                warnings_file=output / f'{name}.warnings.json',
            )
            save_outputs(errors, name, output_dir=output,
                         meta={**cfg, 'setting_id': setting_id, 'iter': args.repetitions, 'g_seed': seed,
                               'bayes_auc': calculate_bayes_auc(cfg)})
    else:
        # Use the original three real-data loaders and per-class sample sizes.
        datasets = [
            ('case_study_1_echo_notes', load_echo_data),
            ('case_study_2_blood_glucose_management_1', load_glucose_data),
            ('case_study_3_blood_gas_oximetry', load_blood_gas_data),
        ]
        sample_sizes = {'n1_tr': 250, 'n2_tr': 250, 'n1_te': 500, 'n2_te': 500}
        for dataset, loader in datasets:
            X, y, groups = loader(config.REAL_DATA_PATH)
            name = f'realdata_{dataset}'
            errors = evaluate_on_real_data(
                X, y, sample_sizes, iter=args.repetitions, g_seed=seed,
                warnings_file=output / f'{name}.warnings.json',
            )
            save_outputs(errors, name, output_dir=output, meta={
                'dataset': dataset, 'n': sample_sizes, 'iter': args.repetitions,
                'g_seed': seed, 'n_features': int(X.shape[1]), 'n_samples': int(X.shape[0]),
            })
    print(f'Results saved in {output}')


if __name__ == '__main__':
    main()
