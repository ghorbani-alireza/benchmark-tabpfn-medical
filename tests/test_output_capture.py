"""Small workflow checks, not an evaluation of the real models.

Run with: python -m unittest discover -s tests
Every function named test_* is a separate check run by unittest. The helper
functions below supply mock models so tests finish quickly without credentials,
checkpoint downloads, or medical datasets.
"""
import ast
import contextlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

# Load the analysis files in an isolated package so importing them does not
# initialize the real benchmark environment or create its output directories.
SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'benchmark'
PACKAGE = '_benchmark_output_test'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package
config = types.ModuleType(f'{PACKAGE}.config')
config.SIM_RESULTS_DIR = Path('/tmp/unused-benchmark-output')
config.REAL_RESULTS_DIR = config.SIM_RESULTS_DIR
sys.modules[config.__name__] = config


def load_module(name):
    """Load one source file under the temporary test-package name."""
    spec = importlib.util.spec_from_file_location(f'{PACKAGE}.{name}', SOURCE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_module('utils')
settings = load_module('settings')
models = types.ModuleType(f'{PACKAGE}.model_definitions')
model_names = (
    'run_linear_sparse_logistic_regression', 'run_sparse_logistic_regression',
    'run_random_forest', 'run_xgboost', 'run_catboost', 'run_qda', 'run_quadro',
    'run_tabpfn_v25', 'run_tabpfn_v26', 'run_tabpfn_v3', 'run_tabpfn_v35',
    'run_exaone', 'run_tabicl',
)


def mock_model(name):
    """Make a tiny substitute that prints, warns, and returns example predictions."""
    def run(X_train, X_test, y_train, y_test, *args, **kwargs):
        """Exercise output capture; the returned measures are test placeholders."""
        print(f'{name}: checkpoint message')
        print(f'{name}: stderr message', file=sys.stderr)
        warnings.warn(f'{name}: warning', UserWarning)
        logging.getLogger('tabicl').warning(f'{name}: logging warning')
        proba = np.where(y_test == 1, 0.8, 0.2)
        return 1., 1., 0., np.array([0, 1]), np.array([0, 1]), proba, .04, .01, .02, .03, .04
    run.__name__ = name
    return run


# Replace every model runner before loading the simulation/resampling code.
for name in model_names:
    setattr(models, name, mock_model(name))
sys.modules[models.__name__] = models
simulation = load_module('simulation_core')
resampling = load_module('resampling_core')


class OutputTests(unittest.TestCase):
    """Check output handling, model selection, and saved-array integrity."""
    def check_run(self, parallel=False, real=False, include_exaone=True, repetitions=2, progress_label=None, terminal_width=None):
        """Run the chosen workflow with mock models and verify its output files."""
        # All files go in a temporary folder that is deleted after the test.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.warnings.json'
            stdout, stderr = io.StringIO(), io.StringIO()
            old_hook = warnings.showwarning
            old_handlers = logging.getLogger().handlers[:]
            # These workflow checks use serial task execution with mock models.
            # Real process isolation and logging have separate integration tests.
            def serial_tasks(**options):
                return lambda tasks: (fn(*args, **kwargs) for fn, args, kwargs in tasks)

            module = resampling if real else simulation
            with patch.object(module, 'Parallel', side_effect=serial_tasks), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                common = dict(n={'n1_tr': 10, 'n2_tr': 10, 'n1_te': 4, 'n2_te': 4},
                              iter=repetitions, warnings_file=path, progress_label=progress_label)
                if real:
                    result = resampling.evaluate_on_real_data(
                        X=np.arange(120).reshape(60, 2), y=np.tile([0, 1], 30),
                        include_exaone=include_exaone, **common)
                else:
                    fn = simulation.evaluate_on_sim_data if parallel else simulation.simulation
                    result = fn(mu={'mu1': np.zeros(2), 'mu2': np.ones(2)},
                                sigma1=np.eye(2), sigma2=np.eye(2),
                                include_exaone=include_exaone, **common)
            # Warnings belong in the JSON file; only progress belongs on screen.
            records = json.loads(path.read_text())
            count = len(model_names) if include_exaone else len(model_names)-1
            self.assertEqual(len(records), count * repetitions)
            self.assertEqual(stdout.getvalue(), '')
            self.assertIn('100%', stderr.getvalue())
            expected_label = progress_label or ('Real-data resampling' if real else
                                               'Simulated-data iterations' if parallel else 'Sim Normal Iterations')
            self.assertIn(expected_label, stderr.getvalue())
            if terminal_width is not None:
                # Each carriage return must overwrite one terminal row, even
                # though capture_output has redirected sys.stderr to a log.
                frames = stderr.getvalue().split('\r')
                self.assertGreater(len(frames), 1)
                for frame in frames:
                    self.assertLessEqual(len(frame.rstrip('\n')), terminal_width)
            self.assertNotIn('checkpoint message', stderr.getvalue())
            self.assertNotIn('warning', stderr.getvalue())
            # Model prints and logging messages must be retained in the log file.
            log = path.with_suffix('.log').read_text()
            for name in model_names:
                if name == 'run_exaone' and not include_exaone:
                    self.assertNotIn(name, log)
                else:
                    self.assertIn(f'{name}: checkpoint message', log)
                    self.assertIn(f'{name}: stderr message', log)
                    self.assertIn(f'{name}: logging warning', log)
            errors = result if parallel or real else result[0]
            self.assertEqual(len(errors['y_test_all']), repetitions)
            self.assertEqual('EXAONE_AUC' in errors, include_exaone)
            for column in errors:
                if column.endswith('_AUC'):
                    prefix = column.removesuffix('_AUC')
                    for metric, expected in (
                        ('SetupTime', .01), ('TuneTime', .02),
                        ('FitTime', .03), ('PredictTime', .04),
                    ):
                        self.assertEqual(errors[f'{prefix}_{metric}'], [expected] * repetitions)
            if not parallel and not real:
                self.assertEqual(result[1], records)
            # The analysis must restore the caller's normal warning/logging settings.
            self.assertIs(warnings.showwarning, old_hook)
            self.assertEqual(logging.getLogger().handlers, old_handlers)

    def test_sequential(self):
        """Check one-at-a-time simulation output capture for every model."""
        self.check_run(progress_label='Setting 1 simulation iterations')

    def test_parallel(self):
        """Check that messages from parallel simulation workers are saved quietly."""
        self.check_run(parallel=True, progress_label='Setting 7+ sensitivity iterations')

    def test_real_resampling(self):
        """Check output capture when resampling existing rows with mock models."""
        self.check_run(real=True, progress_label='Echo resampling iterations')

    def test_progress_fits_terminal_while_stderr_is_redirected(self):
        """Long labels must not make the progress bar wrap onto new rows."""
        for width, parallel, real, label in (
            (60, False, False, 'Setting 1 simulation iterations'),
            (60, True, False, 'Setting 7+ sensitivity iterations'),
            (60, False, True, 'Blood gas resampling iterations'),
            (80, False, False, 'Setting 1 simulation iterations'),
            (80, True, False, 'Setting 7+ sensitivity iterations'),
            (80, False, True, 'Blood gas resampling iterations'),
        ):
            with self.subTest(width=width, parallel=parallel, real=real), patch(
                'tqdm.std._screen_shape_wrapper',
                return_value=lambda stream: (width, 24),
            ):
                self.check_run(parallel=parallel, real=real,
                               progress_label=label, terminal_width=width)

    def test_five_repetition_pilot(self):
        """Check five repetitions and confirm that EXAONE is never called."""
        self.check_run(include_exaone=False, repetitions=5)

    def test_parallel_excludes_exaone(self):
        """Check that the parallel simulation also supports excluding EXAONE."""
        self.check_run(parallel=True, include_exaone=False)

    def test_notebook_pilot(self):
        """Run the actual pilot cell with mock models, checking both saving choices."""
        notebook = json.loads((SOURCE.parents[1] / 'notebooks/run_analysis.ipynb').read_text())
        pilot = next(''.join(cell['source']) for cell in notebook['cells']
                     if cell['cell_type'] == 'code' and 'pilot_bayes =' in ''.join(cell['source']))
        call = next(node for node in ast.walk(ast.parse(pilot))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'simulation')
        options = {item.arg: ast.literal_eval(item.value) for item in call.keywords
                   if item.arg in ('iter', 'include_exaone')}
        for save_output in [False, True]:
            with self.subTest(save_output=save_output), tempfile.TemporaryDirectory() as temporary:
                previous_dir = config.SIM_RESULTS_DIR
                config.SIM_RESULTS_DIR = Path(temporary)
                namespace = {
                    'SIM_RESULTS_DIR': Path(temporary),
                    'SAVE_PILOT_OUTPUT': save_output,
                    'SETTINGS': {2: {
                        'n': {'n1_tr': 10, 'n2_tr': 10, 'n1_te': 4, 'n2_te': 4},
                        'mu': {'mu1': np.zeros(2), 'mu2': np.ones(2)},
                        'sigma1': np.eye(2), 'sigma2': np.eye(2), 'dist': 'Normal',
                    }},
                    'simulation': simulation.simulation, 'save_outputs': utils.save_outputs,
                    'calculate_bayes_auc': settings.calculate_bayes_auc,
                }
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(pilot, 'notebook-pilot', 'exec'), namespace)
                    self.assertEqual(len(namespace['errors']['y_test_all']), options['iter'])
                    self.assertEqual('EXAONE_AUC' in namespace['errors'], options['include_exaone'])
                    result_file = Path(temporary) / 'pilot_setting2/setting2_pilot.csv'
                    self.assertEqual(result_file.is_file(), save_output)
                finally:
                    config.SIM_RESULTS_DIR = previous_dir

    def test_no_warnings_and_failure(self):
        """Check silent success and restoration/saved warnings after a model error."""
        with tempfile.TemporaryDirectory() as temporary:
            stdout, stderr = io.StringIO(), io.StringIO()
            path = Path(temporary)/'quiet.json'
            old_hook = warnings.showwarning
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with utils.capture_output(path, temporary) as (records, progress):
                    print('ordinary message')
                    self.assertIs(progress, stderr)
                self.assertEqual(records, [])
                failing = Path(temporary)/'failed.json'
                with self.assertRaisesRegex(ValueError, 'model failed'):
                    with utils.capture_output(failing, temporary):
                        warnings.warn('before failure')
                        raise ValueError('model failed')
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            self.assertIs(warnings.showwarning, old_hook)
            self.assertEqual(stdout.getvalue()+stderr.getvalue(), '')
            self.assertEqual(json.loads(path.read_text()), [])
            self.assertEqual(len(json.loads(failing.read_text())), 1)

    def test_full_array_round_trip(self):
        """Check that a long prediction array survives saving and loading without losing values."""
        with tempfile.TemporaryDirectory() as temporary:
            original = np.linspace(0, 1, 1200)
            path = utils.save_outputs({'Example_Probas': [original]}, 'results', output_dir=temporary)
            loaded = utils.fix_list_columns(pd.read_csv(path))
            np.testing.assert_array_equal(loaded['Example_Probas'][0], original)
            np.testing.assert_array_equal(utils._parse_array('[0.1 0.2]'), [.1, .2])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                utils._parse_array('[0.1 ... 0.9]')


if __name__ == '__main__':
    unittest.main()
