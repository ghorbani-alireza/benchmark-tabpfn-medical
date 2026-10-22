"""Run the pilot, simulations, sensitivity analysis, resampling, or report saved results."""

import argparse
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=['pilot', 'simulation', 'sensitivity', 'resampling', 'report'])
    parser.add_argument('--repetitions', type=int, help='Number of Monte Carlo repetitions')
    parser.add_argument('--seed', type=int, help='Random seed; default 2025 (sensitivity: 12025)')
    parser.add_argument('--output', type=Path, help='New directory for this analysis')
    parser.add_argument('--results', type=Path, help='Saved result directory to report')
    parser.add_argument('--baseline', type=Path, help='Baseline results for the sensitivity sample-size change table')
    parser.add_argument('--no-bayes-auc', action='store_true', help='Hide the Bayes AUC line in reports')
    parser.add_argument('--no-bayes-roc', action='store_true', help='Hide the Bayes ROC curve in reports')
    parser.add_argument('--jobs', type=int, default=10, help='Worker processes for sensitivity/resampling (default: 10)')
    parser.add_argument('--settings', nargs='+', metavar='ID',
                        help='Setting IDs (pilot default: 2; simulation/sensitivity default: all)')
    parser.add_argument('--datasets', nargs='+', choices=['echo', 'glucose', 'blood_gas'],
                        help='Resampling datasets (default: all)')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.settings is not None and args.analysis not in ('pilot', 'simulation', 'sensitivity'):
        parser.error('--settings is only available for pilot, simulation and sensitivity')
    if args.datasets is not None and args.analysis != 'resampling':
        parser.error('--datasets is only available for resampling')
    if args.baseline is not None and args.analysis != 'report':
        parser.error('--baseline is only available for report')
    if args.analysis == 'pilot' and args.repetitions is None:
        args.repetitions = 20
    if args.analysis == 'report':
        if args.results is None:
            parser.error('report requires --results')
    elif args.repetitions is None or args.repetitions < 1:
        parser.error('choose a positive --repetitions count')

    # Use a plotting backend that saves figures without opening windows.
    import matplotlib
    matplotlib.use('Agg')
    from benchmark import (
        SENSITIVITY,
        SETTINGS,
        calculate_bayes_auc,
        config,
        evaluate_on_real_data,
        load_blood_gas_data,
        load_echo_data,
        load_glucose_data,
        save_outputs,
        simulation,
    )
    from benchmark.simulation_core import evaluate_on_sim_data

    # Validate selections before creating an output folder or running models.
    if args.analysis in ('pilot', 'simulation', 'sensitivity'):
        settings = SENSITIVITY if args.analysis == 'sensitivity' else SETTINGS
        if args.analysis == 'pilot' and args.settings is None:
            settings = {2: settings[2]}
        if args.settings is not None:
            setting_ids = {str(setting_id): setting_id for setting_id in settings}
            unknown = [value for value in args.settings if value not in setting_ids]
            if unknown:
                parser.error(f"unknown --settings ID(s): {', '.join(unknown)}; "
                             f"choose from: {', '.join(setting_ids)}")
            settings = {setting_ids[value]: settings[setting_ids[value]] for value in args.settings}
    elif args.analysis == 'resampling':
        available_datasets = {
            'echo': ('case_study_1_echo_notes', load_echo_data),
            'glucose': ('case_study_2_blood_glucose_management_1', load_glucose_data),
            'blood_gas': ('case_study_3_blood_gas_oximetry', load_blood_gas_data),
        }
        selected = dict.fromkeys(args.datasets if args.datasets is not None else available_datasets)
        datasets = [(name, *available_datasets[name]) for name in selected]

    # Report existing results without loading data or fitting models.
    if args.analysis == 'report':
        from benchmark.plot_functions import (
            plot_boxplots_grid,
            plot_boxplots_grid_lands,
            plot_brier_vs_auc_grid,
            plot_calibration_grid,
            plot_roc_curves_grid,
            plot_runtime_barplot_simple,
        )
        from benchmark.table_functions import (
            METRICS, build_measures_table, build_gain_table,
            build_best_methods_table, build_effect_table,
        )
        from benchmark.utils import discover_sim, load_fixed, read_bayes_auc, read_labels
        files = {path.stem: path for path in sorted(args.results.glob('*.csv'))}
        if not files:
            parser.error('no result CSV files found in --results directory')
        if args.baseline is not None:
            baseline_files = discover_sim(args.baseline)
            sensitivity_files = discover_sim(args.results, variant='sensitivity')
            if not baseline_files or not sensitivity_files:
                parser.error('--baseline needs baseline simulation CSVs and --results needs sensitivity CSVs')
            if not set(baseline_files).intersection(sensitivity_files):
                parser.error('baseline and sensitivity directories have no matching settings')
        results = load_fixed(files)
        labels = read_labels({name: str(path) for name, path in files.items()})
        bayes = read_bayes_auc(files) if not (args.no_bayes_auc and args.no_bayes_roc) else {}
        names = list(results)
        table_dir = args.results / 'tables'
        plot_dir = args.results / 'plots'
        table_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        build_measures_table(results, filename='measures.csv', tables_dir=table_dir)
        build_best_methods_table(results, filename='best_methods.csv', tables_dir=table_dir)
        build_effect_table(results, filename='best_vs_best.csv', tables_dir=table_dir)
        build_effect_table(results, filename='model_effects.csv', tables_dir=table_dir, all_models=True)
        if args.baseline is not None:
            build_gain_table(
                load_fixed(baseline_files), load_fixed(sensitivity_files), metrics=METRICS,
                filename='sample_size_changes.csv', tables_dir=table_dir,
            )
        plots = str(plot_dir) + '/'
        plot_boxplots_grid(results, names, labels=labels, metric='AUC', dir=plots, save_path='auc_box.png', bayes_auc=bayes, show_bayes_auc=not args.no_bayes_auc)
        plot_boxplots_grid_lands(results, names, labels=labels, metric='AUC', dir=plots, save_path='auc_box_landscape.png', bayes_auc=bayes, show_bayes_auc=not args.no_bayes_auc)
        plot_roc_curves_grid(results, names, labels=labels, dir=plots, save_path='roc.png', bayes_auc=bayes, show_bayes_roc=not args.no_bayes_roc)
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
    seed = args.seed if args.seed is not None else (12025 if args.analysis == 'sensitivity' else 2025)

    # The pilot and main simulation are sequential; sensitivity uses parallel repetitions.
    if args.analysis in ('pilot', 'simulation', 'sensitivity'):
        runner = evaluate_on_sim_data if args.analysis == 'sensitivity' else simulation
        for setting_id, cfg in settings.items():
            suffix = '_sensitivity' if args.analysis == 'sensitivity' else ''
            name = (f'setting{setting_id}_pilot' if args.analysis == 'pilot'
                    else f'simulation_setting{setting_id}{suffix}')
            worker_options = {'n_jobs': args.jobs} if args.analysis == 'sensitivity' else {}
            result = runner(
                n=cfg['n'], mu=cfg['mu'], sigma1=cfg['sigma1'], sigma2=cfg['sigma2'],
                dist=cfg['dist'], df=cfg.get('df'), iter=args.repetitions, g_seed=seed,
                include_exaone=args.analysis == 'pilot', **worker_options,
                progress_label=f'Setting {setting_id} {args.analysis} iterations',
                warnings_file=output / 'warnings' / f'{name}.warnings.json',
            )
            # simulation returns (results, warnings); the parallel runner returns results.
            if args.analysis in ('pilot', 'simulation'):
                errors, _warns = result
            else:
                errors = result
            save_outputs(errors, name, output_dir=output,
                         meta={**cfg, 'setting_id': setting_id, 'iter': args.repetitions, 'g_seed': seed,
                               'include_exaone': args.analysis == 'pilot',
                               'execution': 'parallel' if args.analysis == 'sensitivity' else 'sequential',
                               'workers': args.jobs if args.analysis == 'sensitivity' else 1,
                               'threads_per_worker': 1,
                               'bayes_auc': calculate_bayes_auc(cfg)})
    else:
        # Use the selected real-data loaders and original per-class sample sizes.
        sample_sizes = {'n1_tr': 250, 'n2_tr': 250, 'n1_te': 500, 'n2_te': 500}
        for dataset_key, dataset, loader in datasets:
            X, y, _groups = loader(config.REAL_DATA_PATH)
            name = f'realdata_{dataset}'
            errors = evaluate_on_real_data(
                X, y, sample_sizes, iter=args.repetitions, g_seed=seed,
                n_jobs=args.jobs, include_exaone=False,
                progress_label=f"{dataset_key.replace('_', ' ').capitalize()} resampling iterations",
                warnings_file=output / 'warnings' / f'{name}.warnings.json',
            )
            save_outputs(errors, name, output_dir=output, meta={
                'dataset': dataset, 'n': sample_sizes, 'iter': args.repetitions,
                'g_seed': seed, 'n_features': int(X.shape[1]), 'n_samples': int(X.shape[0]),
                'include_exaone': False, 'execution': 'parallel',
                'workers': args.jobs, 'threads_per_worker': 1,
            })
    print(f'Results saved in {output}')


if __name__ == '__main__':
    main()
