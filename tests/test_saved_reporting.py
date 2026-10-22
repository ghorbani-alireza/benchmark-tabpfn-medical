"""Exercise terminal and fresh-notebook reports without fitting any model."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark
from benchmark import config
from benchmark.table_functions import METRICS
from benchmark.utils import save_outputs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_saved_reporting_cli', ROOT / 'scripts/run_analysis.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def example_results(directory, stem, shift=0):
    """Save small artificial test results and their population reference."""
    outcomes = np.array([0, 1, 0, 1])
    errors = {'y_test_all': [outcomes] * 3}
    for index, method in enumerate(['L-SLR', 'QDA', 'TabICL-V1', 'TabICL-V2']):
        for metric in METRICS:
            values = [.1 + .01 * index] * 3
            if metric == 'AUC':
                values = [.65 + .03 * index + .01 * run + shift for run in range(3)]
            elif metric == 'CalSlope':
                values = [.9 + .02 * index] * 3
            errors[f'{method}_{metric}'] = values
        errors[f'{method}_FPR'] = [[0., .2, 1.]] * 3
        errors[f'{method}_TPR'] = [[0., .7 + .03 * index, 1.]] * 3
        errors[f'{method}_Probas'] = [[.1, .8, .2, .9]] * 3
    save_outputs(errors, stem, output_dir=directory, meta={
        'setting_id': 1, 'label': 'Test results', 'iter': 3,
        'bayes_auc': {'auc': .9, 'mc_se': .001, 'fpr': [0., .2, 1.], 'tpr': [0., .95, 1.]},
    })


class SavedReportingTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_terminal_reports_include_rankings_effects_and_sample_size_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'main'
            sens = Path(directory) / 'sensitivity'
            example_results(base, 'simulation_setting1')
            example_results(sens, 'simulation_setting1_sensitivity', shift=.05)
            with patch.object(benchmark, 'simulation', side_effect=AssertionError('Unexpected fit')), patch.object(
                benchmark, 'evaluate_on_real_data', side_effect=AssertionError('Unexpected resampling'),
            ), contextlib.redirect_stdout(io.StringIO()):
                cli.main(['report', '--results', str(sens), '--baseline', str(base)])
            for filename in ['measures.csv', 'best_methods.csv', 'best_vs_best.csv',
                             'model_effects.csv', 'sample_size_changes.csv']:
                self.assertTrue((sens / 'tables' / filename).is_file(), filename)
            for filename in ['auc_box.png', 'auc_box_landscape.png', 'roc.png',
                             'calibration.png', 'brier_auc.png', 'runtime.png']:
                self.assertTrue((sens / 'plots' / filename).is_file(), filename)
            changes = pd.read_csv(sens / 'tables/sample_size_changes.csv', index_col=['Setting', 'Metric'])
            self.assertAlmostEqual(changes.loc[('S1', 'AUC'), 'TabICL-V1'], .05)
            effects = pd.read_csv(sens / 'tables/best_vs_best.csv')
            auc = effects[effects['Metric'] == 'AUC'].iloc[0]
            self.assertEqual(auc['Classical method'], 'QDA')
            self.assertEqual(auc['Foundation method'], 'TabICL-V2')
            self.assertEqual(auc['Paired runs'], 3)

    def test_notebook_loads_saved_tables_and_plots_in_an_empty_namespace(self):
        notebook = json.loads((ROOT / 'notebooks/run_analysis.ipynb').read_text())
        report = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and 'RESULTS_DIR =' in ''.join(cell['source']))
        plot_cells = [''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('plot_')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / 'pilot_setting2'
            example_results(saved, 'setting2_pilot')
            namespace = {}
            with patch.object(config, 'SIM_RESULTS_DIR', root), patch.object(config, 'REAL_RESULTS_DIR', root), patch.object(
                benchmark, 'simulation', side_effect=AssertionError('Unexpected pilot'),
            ), patch('IPython.display.display') as display, contextlib.redirect_stdout(io.StringIO()):
                exec(compile(report, 'notebook-saved-results', 'exec'), namespace)
                for source in plot_cells:
                    exec(compile(source, 'notebook-saved-plot', 'exec'), namespace)
            self.assertNotIn('errors', namespace)
            self.assertEqual(namespace['bayes']['setting2_pilot']['auc'], .9)
            self.assertIn('TabICL-V1_AUC', namespace['results']['setting2_pilot'])
            self.assertTrue((saved / 'tables/best_vs_best.csv').is_file())
            self.assertTrue((saved / 'plots/runtime.png').is_file())
            self.assertGreaterEqual(display.call_count, 5)
            for call in display.call_args_list:
                self.assertIn('<table', call.args[0].to_html())


if __name__ == '__main__':
    unittest.main()
