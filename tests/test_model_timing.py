"""Verify timing arithmetic and boundaries without relying on sleep durations."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split

from benchmark import model_definitions as models


class ModelTimingTests(unittest.TestCase):
    def test_separate_tune_fit_and_prediction_for_every_classical_runner(self):
        rng = np.random.default_rng(41)
        y = np.tile([0, 1], 30)
        X = rng.normal(size=(60, 2)) + y[:, None]
        X_sub, X_val, y_sub, y_val = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=41,
        )

        for runner in (
            models.run_linear_sparse_logistic_regression,
            models.run_sparse_logistic_regression,
            models.run_random_forest, models.run_xgboost,
            models.run_catboost, models.run_qda, models.run_quadro,
        ):
            with self.subTest(runner=runner.__name__):
                # setup = 12-10; tune = 25-20; fit = 34-30; predict = 47-40.
                with patch.object(
                    models, 'perf_counter', side_effect=[10., 12., 20., 25., 30., 34., 40., 47.],
                ):
                    result = runner(
                        X, X_val, y, y_val, X_sub, y_sub, X_val, y_val,
                        random_state=41,
                    )
                self.assertEqual(result[7:], (2., 5., 4., 7.))
                self.assertEqual(len(result), 11)

    def test_foundation_fit_and_prediction_exclude_metrics_and_keep_setup_time(self):
        class FoundationModel:
            def __init__(self, **kwargs):
                self.classes_ = np.array([0, 1])

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return np.array([[.8, .2], [.3, .7]])

        X = np.array([[0.], [1.]])
        y = np.array([0, 1])
        with patch.object(models, 'perf_counter', side_effect=[10., 12., 20., 25., 30., 37.]):
            result = models._run_tabpfn_generic(
                X, X, y, y, FoundationModel, {}, 41,
                setup_time=4.,
            )
        self.assertEqual(result[7:], (6., 0., 5., 7.))
        self.assertEqual(len(result), 11)


if __name__ == '__main__':
    unittest.main()
