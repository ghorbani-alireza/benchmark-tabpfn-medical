"""Exercise CLI selection without importing model libraries or loading datasets."""

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_analysis.py'
spec = importlib.util.spec_from_file_location('_run_analysis_test', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class AnalysisSelectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.output = Path(folder.name) / 'results'
        self.benchmark = types.ModuleType('benchmark')
        self.benchmark.SETTINGS = {
            sid: {'n': {'setting': sid}, 'mu': {}, 'sigma1': None, 'sigma2': None,
                  'dist': 'Normal', 'label': f'Setting {sid}'}
            for sid in range(1, 9)
        }
        self.benchmark.SENSITIVITY = {
            **self.benchmark.SETTINGS,
            '7+': {'n': {'setting': '7+'}, 'mu': {}, 'sigma1': None, 'sigma2': None,
                   'dist': 't', 'df': 5, 'label': 'Setting 7+'},
        }
        self.benchmark.config = types.SimpleNamespace(
            SIM_RESULTS_DIR=self.output.parent,
            REAL_RESULTS_DIR=self.output.parent,
            REAL_DATA_PATH=self.output.parent / 'data',
        )
        self.benchmark.simulation = Mock(return_value=({'result': 'main'}, []))
        self.benchmark.evaluate_on_real_data = Mock(return_value={'result': 'real'})
        self.benchmark.save_outputs = Mock()
        self.benchmark.calculate_bayes_auc = Mock(return_value={'auc': .8})
        for name in ('load_echo_data', 'load_glucose_data', 'load_blood_gas_data'):
            setattr(self.benchmark, name, Mock(return_value=(
                types.SimpleNamespace(shape=(20, 2)), list(range(20)), None,
            )))
        self.core = types.ModuleType('benchmark.simulation_core')
        self.core.evaluate_on_sim_data = Mock(return_value={'result': 'sensitivity'})
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.use = Mock()
        modules = patch.dict(sys.modules, {
            'benchmark': self.benchmark,
            'benchmark.simulation_core': self.core,
            'matplotlib': matplotlib,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def run_cli(self, analysis, *options):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main([analysis, '--repetitions', '3', '--output', str(self.output), *options])

    def test_pilot_defaults_to_twenty_sequential_setting_two_runs_with_exaone(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(['pilot', '--output', str(self.output)])
        options = self.benchmark.simulation.call_args.kwargs
        self.assertEqual(options['n']['setting'], 2)
        self.assertEqual(options['iter'], 20)
        self.assertEqual(options['g_seed'], 2025)
        self.assertTrue(options['include_exaone'])
        self.assertNotIn('n_jobs', options)
        self.core.evaluate_on_sim_data.assert_not_called()
        save = self.benchmark.save_outputs.call_args
        self.assertEqual(save.args, ({'result': 'main'}, 'setting2_pilot'))
        self.assertEqual(save.kwargs['meta']['execution'], 'sequential')
        self.assertEqual(save.kwargs['meta']['workers'], 1)
        self.assertTrue(save.kwargs['meta']['include_exaone'])

    def test_pilot_accepts_overrides_without_parallel_calls(self):
        self.run_cli('pilot', '--settings', '3', '--seed', '50', '--jobs', '2')
        options = self.benchmark.simulation.call_args.kwargs
        self.assertEqual(options['n']['setting'], 3)
        self.assertEqual(options['iter'], 3)
        self.assertEqual(options['g_seed'], 50)
        self.assertTrue(options['include_exaone'])
        self.assertNotIn('n_jobs', options)
        self.core.evaluate_on_sim_data.assert_not_called()

    def test_main_subset_preserves_ids_order_and_saves_only_selected_results(self):
        self.run_cli('simulation', '--settings', '5', '2', '5')
        self.assertEqual(
            [call.kwargs['n']['setting'] for call in self.benchmark.simulation.call_args_list],
            [5, 2],
        )
        self.assertEqual(
            [call.kwargs['progress_label'] for call in self.benchmark.simulation.call_args_list],
            ['Setting 5 simulation iterations', 'Setting 2 simulation iterations'],
        )
        for call in self.benchmark.simulation.call_args_list:
            self.assertEqual(call.kwargs['warnings_file'].parent, self.output / 'warnings')
        saves = self.benchmark.save_outputs.call_args_list
        self.assertEqual([call.args[1] for call in saves],
                         ['simulation_setting5', 'simulation_setting2'])
        for call in saves:
            self.assertEqual(call.args[0], {'result': 'main'})
            self.assertEqual(call.kwargs['output_dir'], self.output)
            self.assertEqual(call.kwargs['meta']['iter'], 3)
            self.assertEqual(call.kwargs['meta']['g_seed'], 2025)
            self.assertIsInstance(call.kwargs['meta']['setting_id'], int)
        self.core.evaluate_on_sim_data.assert_not_called()

    def test_sensitivity_extra_setting_and_worker_options(self):
        self.run_cli('sensitivity', '--settings', '7+', '--jobs', '2')
        self.core.evaluate_on_sim_data.assert_called_once()
        options = self.core.evaluate_on_sim_data.call_args.kwargs
        self.assertEqual(options['n']['setting'], '7+')
        self.assertEqual(options['df'], 5)
        self.assertEqual(options['iter'], 3)
        self.assertEqual(options['g_seed'], 12025)
        self.assertEqual(options['n_jobs'], 2)
        self.assertEqual(options['progress_label'], 'Setting 7+ sensitivity iterations')
        self.assertEqual(options['warnings_file'],
                         self.output / 'warnings' / 'simulation_setting7+_sensitivity.warnings.json')
        save = self.benchmark.save_outputs.call_args
        self.assertEqual(save.args, ({'result': 'sensitivity'}, 'simulation_setting7+_sensitivity'))
        self.assertEqual(save.kwargs['meta']['setting_id'], '7+')
        self.benchmark.simulation.assert_not_called()

    def test_resampling_loads_and_saves_only_selected_datasets(self):
        self.run_cli('resampling', '--datasets', 'blood_gas', 'echo', 'blood_gas', '--jobs', '2')
        self.benchmark.load_glucose_data.assert_not_called()
        self.benchmark.load_echo_data.assert_called_once()
        self.benchmark.load_blood_gas_data.assert_called_once()
        self.assertEqual(self.benchmark.evaluate_on_real_data.call_count, 2)
        self.assertEqual(
            [call.kwargs['progress_label'] for call in self.benchmark.evaluate_on_real_data.call_args_list],
            ['Blood gas resampling iterations', 'Echo resampling iterations'],
        )
        saves = self.benchmark.save_outputs.call_args_list
        self.assertEqual([call.args[1] for call in saves], [
            'realdata_case_study_3_blood_gas_oximetry', 'realdata_case_study_1_echo_notes',
        ])
        for call in self.benchmark.evaluate_on_real_data.call_args_list:
            self.assertEqual(call.kwargs['warnings_file'].parent, self.output / 'warnings')
            self.assertEqual(call.kwargs['n_jobs'], 2)
            self.assertEqual(call.kwargs['iter'], 3)
            self.assertEqual(call.kwargs['g_seed'], 2025)

    def test_omitting_selection_runs_all(self):
        for analysis, expected in (('simulation', 8), ('sensitivity', 9), ('resampling', 3)):
            with self.subTest(analysis=analysis):
                self.output = self.output.parent / analysis
                self.benchmark.save_outputs.reset_mock()
                self.run_cli(analysis)
                self.assertEqual(self.benchmark.save_outputs.call_count, expected)

    def test_invalid_selection_fails_before_creating_output_or_running_models(self):
        cases = [
            ('simulation', ['--settings', '2', '99'], 'unknown --settings'),
            ('simulation', ['--settings', '7+'], 'unknown --settings'),
            ('sensitivity', ['--settings', '0'], 'unknown --settings'),
            ('resampling', ['--datasets', 'unknown'], 'invalid choice'),
            ('resampling', ['--settings', '2'], '--settings is only available'),
            ('simulation', ['--datasets', 'echo'], '--datasets is only available'),
            ('sensitivity', ['--datasets', 'echo'], '--datasets is only available'),
            ('report', ['--settings', '2'], '--settings is only available'),
            ('report', ['--datasets', 'echo'], '--datasets is only available'),
            ('simulation', ['--settings'], 'expected at least one argument'),
            ('resampling', ['--datasets'], 'expected at least one argument'),
            ('pilot', ['--settings', '7+'], 'unknown --settings'),
            ('pilot', ['--baseline', 'unused'], '--baseline is only available'),
        ]
        for analysis, options, message in cases:
            with self.subTest(analysis=analysis, options=options):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    self.run_cli(analysis, *options)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                self.assertFalse(self.output.exists())
        self.benchmark.simulation.assert_not_called()
        self.core.evaluate_on_sim_data.assert_not_called()
        self.benchmark.evaluate_on_real_data.assert_not_called()
        for name in ('load_echo_data', 'load_glucose_data', 'load_blood_gas_data'):
            getattr(self.benchmark, name).assert_not_called()
        self.benchmark.save_outputs.assert_not_called()


if __name__ == '__main__':
    unittest.main()
