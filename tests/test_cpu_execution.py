"""Check actual worker processes, thread limits and quiet log collection."""
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_info

from benchmark import simulation
from benchmark.simulation_core import evaluate_on_sim_data
from benchmark.resampling_core import evaluate_on_real_data


def example_iteration(*args):
    """Return tiny model-shaped results without downloading or fitting models."""
    import warnings
    pools = threadpool_info()
    print(json.dumps({'pid': os.getpid(), 'torch': torch.get_num_threads(),
                     'interop': torch.get_num_interop_threads(),
                     'pools': [pool['num_threads'] for pool in pools]}))
    print('worker prediction message')
    warnings.warn('example worker warning', UserWarning)
    # Keep each task busy briefly so both worker processes receive work.
    time.sleep(.2)
    y = np.array([0, 1])
    measures = (.01, 0., 1., np.array([0., 1.]), np.array([0., 1.]),
                np.array([.1, .9]), .01, 1., 0.)
    return {**{name: measures for name in (
        'L-SLR', 'SLR', 'RandomForest', 'XGBoost', 'CatBoost', 'Quadro',
        'TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35', 'TabICL-V2',
    )}, 'y_test': y}


def failing_iteration(*args):
    """Exercise a child failure after it has written a message and warning."""
    import warnings
    print('message before worker failure')
    warnings.warn('warning before worker failure', UserWarning)
    raise RuntimeError('example process failure')


class CPUExecutionTests(unittest.TestCase):
    def test_parent_and_default_worker_count(self):
        self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_interop_threads(), 1)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(os.environ[name], '1')
        for pool in threadpool_info():
            self.assertEqual(pool['num_threads'], 1, pool)
        self.assertEqual(inspect.signature(evaluate_on_sim_data).parameters['n_jobs'].default, 10)
        self.assertEqual(inspect.signature(evaluate_on_real_data).parameters['n_jobs'].default, 10)
        self.assertNotIn('n_jobs', inspect.signature(simulation).parameters)

    def test_failed_worker_keeps_private_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'failed.warnings.json'
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch('benchmark.simulation_core._single_sim_iteration', failing_iteration), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, 'example process failure'):
                    evaluate_on_sim_data(
                        n={'n1_tr': 3, 'n2_tr': 3, 'n1_te': 2, 'n2_te': 2},
                        mu={'mu1': np.zeros(2), 'mu2': np.ones(2)},
                        sigma1=np.eye(2), sigma2=np.eye(2), iter=1, n_jobs=2,
                        warnings_file=path,
                    )
            self.assertEqual(stdout.getvalue(), '')
            worker = path.with_suffix('.workers') / 'iteration_1.warnings.json'
            self.assertEqual(json.loads(worker.read_text())[0]['message'], 'warning before worker failure')
            self.assertIn('message before worker failure', worker.with_suffix('.log').read_text())
            self.assertTrue(path.is_file())

    def test_parallel_simulation_and_resampling(self):
        for real in (False, True):
            with self.subTest(real=real), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'run.warnings.json'
                options = dict(n={'n1_tr': 3, 'n2_tr': 3, 'n1_te': 2, 'n2_te': 2},
                               iter=4, n_jobs=2, warnings_file=path)
                target = ('benchmark.resampling_core._single_real_iteration' if real
                          else 'benchmark.simulation_core._single_sim_iteration')
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch(target, example_iteration), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    if real:
                        results = evaluate_on_real_data(np.arange(40).reshape(20, 2),
                                                        np.tile([0, 1], 10), **options)
                    else:
                        results = evaluate_on_sim_data(mu={'mu1': np.zeros(2), 'mu2': np.ones(2)},
                                                       sigma1=np.eye(2), sigma2=np.eye(2), **options)
                self.assertEqual(stdout.getvalue(), '')
                self.assertIn('100%', stderr.getvalue())
                self.assertNotIn('worker prediction message', stderr.getvalue())
                self.assertEqual(len(json.loads(path.read_text())), 4)
                log = path.with_suffix('.log').read_text()
                states = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
                self.assertEqual(len(states), 4)
                self.assertEqual(len({state['pid'] for state in states}), 2)
                for state in states:
                    self.assertNotEqual(state['pid'], os.getpid())
                    self.assertEqual(state['torch'], 1)
                    self.assertEqual(state['interop'], 1)
                    self.assertTrue(all(count == 1 for count in state['pools']))
                self.assertEqual(len(list(path.with_suffix('.workers').glob('*.json'))), 4)
                self.assertEqual(len(results['SLR_AUC']), 4)
                self.assertFalse(any(key.startswith('EXAONE_') for key in results))


if __name__ == '__main__':
    unittest.main()
